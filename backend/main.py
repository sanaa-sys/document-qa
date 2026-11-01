import asyncio
import hashlib
import hmac
import threading
import urllib.error
import urllib.request
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
import json
import os
from datetime import datetime
import re

from pinecone_store import (
    DocumentNotFound,
    count_ready_chunks,
    count_ready_documents,
    delete_chunks,
    delete_document,
    embed_model_name,
    embed_texts as pinecone_embed_texts,
    embedding_dim,
    get_document_by_filename,
    insert_document,
    keyword_chunks,
    list_documents,
    ping_store,
    replace_chunks,
    search_chunks,
    store_uri,
    update_document,
    uses_integrated_embed,
)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

app = FastAPI(title="MedChat API")

# ──────────────────────────────────────────────
# CORS — allow Next.js frontend
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://1-3rd.vercel.app",
        "https://1-3rd.vercel.app/*",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
TEMPERATURE = 0.1
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "768"))
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_log.json")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME") or embed_model_name()
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 200
INGEST_BATCH_SIZE = 50
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.05"))
MATCH_COUNT = int(os.getenv("MATCH_COUNT", "5"))


# WhatsApp Cloud API (Meta). Leave unset to disable the WhatsApp bot.
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "").strip()
WHATSAPP_GRAPH_VERSION = os.getenv("WHATSAPP_GRAPH_VERSION", "v21.0").strip() or "v21.0"
WHATSAPP_MAX_CHARS = 4000
_PROCESSED_WA_IDS = set()
_PROCESSED_WA_MAX = 2000
_WA_GREETING = re.compile(
    r"^(hi|hello|hey|yo|salam|salaam|assalamu[\s\-]?alaikum|"
    r"as[\s\-]?salamu[\s\-]?alaykum|good\s+(morning|afternoon|evening))\b",
    re.IGNORECASE,
)

MODEL = None
MODEL_BACKEND = None  # "pinecone"


def ensure_model():
    """Embeddings come from Pinecone Inference so they match the index dimension."""
    global MODEL, MODEL_BACKEND
    if MODEL_BACKEND == "pinecone":
        return MODEL
    ping_store()
    MODEL = True
    MODEL_BACKEND = "pinecone"
    print(f"✅ Embedding backend: pinecone ({embed_model_name()}, {embedding_dim()}-d)")
    return MODEL


def _warm_embeddings():
    try:
        ensure_model()
        print("✅ Embeddings warmed in background")
    except Exception as e:
        print(f"⚠️ Embedding warmup failed: {e}")


def embed_texts(texts: list, input_type: str = "passage") -> list:
    """Return embedding vectors that match the Pinecone index dimension."""
    ensure_model()
    return pinecone_embed_texts(texts, input_type=input_type)


def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def retrieve_chunks(question: str, top_k: int = MATCH_COUNT) -> list:
    question_vec = []
    if not uses_integrated_embed():
        try:
            question_vec = embed_texts([question], input_type="query")[0]
        except Exception as e:
            print(f"⚠️ Embedding failed: {e}")

    for threshold in (MATCH_THRESHOLD, 0.0):
        try:
            hits = search_chunks(question_vec, top_k, threshold, query_text=question)
        except Exception as e:
            print(f"⚠️ Documents search failed (threshold={threshold}): {e}")
            hits = []
        if hits:
            print(f"✅ Document hits: {len(hits)} (threshold={threshold}, best={hits[0]['score']:.3f})")
            return hits

    try:
        hits = keyword_chunks(question, top_k)
    except Exception as e:
        print(f"⚠️ Keyword fallback failed: {e}")
        return []
    if hits:
        print(f"✅ Keyword fallback hits: {len(hits)}")
        return hits
    print("⚠️ No chunk hits — is the knowledge base empty? Run /ingest.")
    return []


@app.on_event("startup")
async def load_rag():
    """Boot fast: do not load embedding models unless explicitly asked."""
    docs = count_ready_documents()
    chunks = count_ready_chunks()
    if docs is None:
        print("⚠️ Pinecone could not be opened. Set PINECONE_API_KEY (and optional PINECONE_INDEX).")
    else:
        print(f"✅ Pinecone ready: {docs} ready documents, {chunks} chunks (index={store_uri()})")

    if os.getenv("LOAD_EMBEDDING_ON_STARTUP", "").strip().lower() in ("1", "true", "yes"):
        try:
            ensure_model()
        except Exception as e:
            print(f"⚠️ Embedding model load failed: {e}")
    else:
        print("ℹ️ Embeddings will load lazily on first /chat (saves RAM at boot)")

    # Warm embeddings in the background so the first user question is not
    # blocked by model download/load (does not delay Railway health checks).
    disable_warmup = os.getenv("DISABLE_EMBEDDING_WARMUP", "").strip().lower() in (
        "1", "true", "yes",
    )
    if MODEL is None and not disable_warmup:
        threading.Thread(
            target=_warm_embeddings,
            daemon=True,
            name="embed-warmup",
        ).start()

    if os.getenv("INGEST_ON_STARTUP", "").strip().lower() in ("1", "true", "yes"):
        try:
            result = ingest_data_files(force=False)
            print(f"✅ Startup ingest: {result}")
        except Exception as e:
            print(f"⚠️ Startup ingest failed: {e}")


# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float

class FeedbackRequest(BaseModel):
    rating: int
    accuracy: int
    categories: List[str]
    comments: str


# ──────────────────────────────────────────────
# PDF INGEST → documents + chunks
# ──────────────────────────────────────────────
def list_pdf_files() -> list:
    if not os.path.isdir(DATA_DIR):
        return []
    files = []
    for name in sorted(os.listdir(DATA_DIR)):
        if name.lower().endswith(".pdf"):
            files.append(os.path.join(DATA_DIR, name))
    return files


def chunk_text(text: str, max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_chunks(path: str) -> tuple:
    """Return (page_count, rows without document_id/embeddings)."""
    import logging
    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)

    reader = PdfReader(path)
    rows = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        for text in chunk_text(page_text):
            rows.append({
                "page": page_number,
                "text": text,
                "metadata": {"page": page_number},
            })

    # Re-number chunk_index across the whole document
    for i, row in enumerate(rows):
        row["chunk_index"] = i
    return len(reader.pages), rows


def ingest_one_pdf(path: str, force: bool = False) -> dict:
    filename = os.path.basename(path)
    title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip() or filename
    checksum = file_checksum(path)
    existing = get_document_by_filename(filename)

    if existing and not force:
        if existing.get("checksum") == checksum and existing.get("status") == "ready":
            return {
                "filename": filename,
                "status": "skipped",
                "reason": "unchanged checksum",
                "document_id": existing["id"],
                "chunks_upserted": 0,
            }

    if existing:
        document_id = existing["id"]
        update_document(document_id, {
            "title": title,
            "checksum": checksum,
            "status": "processing",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "error": None,
        })
        delete_chunks(document_id)
    else:
        document_id = insert_document({
            "title": title,
            "filename": filename,
            "checksum": checksum,
            "status": "processing",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "metadata": {"path": filename},
        })

    try:
        page_count, rows = extract_pdf_chunks(path)
        if not rows:
            update_document(document_id, {
                "status": "failed",
                "error": "No extractable text",
                "page_count": page_count,
                "chunk_count": 0,
            })
            return {
                "filename": filename,
                "status": "failed",
                "reason": "no extractable text",
                "document_id": document_id,
                "chunks_upserted": 0,
            }

        if uses_integrated_embed():
            embeddings = [[] for _ in rows]
        else:
            embeddings = embed_texts([row["text"] for row in rows])
        chunk_rows = []
        for row, vector in zip(rows, embeddings):
            chunk_rows.append({
                "chunk_index": row["chunk_index"],
                "page": row["page"],
                "text": row["text"],
                "vector": vector,
                "embedding_model": EMBEDDING_MODEL_NAME,
            })

        replace_chunks(
            document_id,
            filename,
            title,
            chunk_rows,
            batch_size=INGEST_BATCH_SIZE,
        )
        update_document(document_id, {
            "status": "ready",
            "page_count": page_count,
            "chunk_count": len(chunk_rows),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "error": None,
        })

        print(f"✅ Ingested {filename}: {len(chunk_rows)} chunks ({page_count} pages)")
        return {
            "filename": filename,
            "status": "ingested",
            "document_id": document_id,
            "chunks_upserted": len(chunk_rows),
            "page_count": page_count,
        }
    except Exception as e:
        update_document(document_id, {
            "status": "failed",
            "error": str(e)[:500],
        })
        raise


def ingest_data_files(force: bool = False) -> dict:
    """Parse PDFs in ./data into the Pinecone documents and chunks namespaces."""
    pdfs = list_pdf_files()
    empty = {
        "ingested_files": [],
        "skipped_files": [],
        "failed_files": [],
        "errors": {},
        "chunks_upserted": 0,
        "documents": None,
        "chunks": None,
    }

    try:
        ping_store()
    except Exception as e:
        print(f"⚠️ Pinecone preflight failed: {e}")
        empty["failed_files"] = [os.path.basename(p) for p in pdfs]
        empty["error"] = str(e)
        empty["errors"] = {name: str(e) for name in empty["failed_files"]}
        return empty

    try:
        ensure_model()
    except Exception as e:
        print(f"⚠️ Embedding preflight failed: {e}")
        empty["failed_files"] = [os.path.basename(p) for p in pdfs]
        empty["error"] = str(e)
        empty["errors"] = {name: str(e) for name in empty["failed_files"]}
        return empty

    if not pdfs:
        empty["documents"] = count_ready_documents()
        empty["chunks"] = count_ready_chunks()
        return empty

    ingested = []
    skipped = []
    failed = []
    errors = {}
    chunks_upserted = 0

    for path in pdfs:
        try:
            result = ingest_one_pdf(path, force=force)
            if result["status"] == "skipped":
                skipped.append(result["filename"])
            elif result["status"] == "failed":
                failed.append(result["filename"])
                if result.get("reason"):
                    errors[result["filename"]] = result["reason"]
            else:
                ingested.append(result["filename"])
                chunks_upserted += result.get("chunks_upserted", 0)
        except Exception as e:
            failed.append(os.path.basename(path))
            errors[os.path.basename(path)] = str(e)
            print(f"⚠️ Ingest failed for {path}: {e}")

    return {
        "ingested_files": ingested,
        "skipped_files": skipped,
        "failed_files": failed,
        "errors": errors,
        "chunks_upserted": chunks_upserted,
        "documents": count_ready_documents(),
        "chunks": count_ready_chunks(),
    }


def strip_think_tags(text: str) -> str:
    """Remove model reasoning tags, including unclosed <think> blocks."""
    if not text:
        return text
    cleaned = re.sub(
        r'<think(?:ing)?>.*?</think(?:ing)?>',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r'<think(?:ing)?>.*$',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r'</?think(?:ing)?>', '', cleaned, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()


SYSTEM_INSTRUCTION = (
    "You are ONE-THIRD, a medical knowledge assistant. Answer ONLY from the "
    "provided context. Never invent remedies, doses, or drug names.\n\n"
    "TREATMENT / SELF-CARE QUESTIONS (symptoms, conditions, how to treat, "
    "relieve, or manage something):\n"
    "Write the entire answer under these three Markdown headings, in this order:\n"
    "## Effective Home Remedies\n"
    "## Over-the-Counter Medications\n"
    "## What Not to Do\n\n"
    "Under each heading use short bullet points. Cite a page when a fact comes "
    "from the context, e.g. (p. 49). If the context has nothing for a heading, "
    "write one bullet: Not mentioned in the available documents.\n\n"
    "OTHER QUESTIONS (benefits, explanations, what is X):\n"
    "Answer in short bullets or a brief paragraph. Do not force those three headings.\n\n"
    "If the context is insufficient, say so plainly. Never include <think> tags "
    "or chain-of-thought."
)


def _format_context(chunks: list) -> str:
    parts = [
        f"[{c.get('title') or c.get('source') or 'Document'} · p.{c['page']}]: {c['text']}"
        for c in chunks[:5]
    ]
    return "\n\n---\n\n".join(parts)


def _sources_from_chunks(chunks: list) -> list:
    return [
        {
            "page": c["page"],
            "source": os.path.basename(str(c.get("source") or "Document")),
            "title": c.get("title") or c.get("source") or "Document",
            "document_id": c.get("document_id"),
            "score": round(float(c.get("score") or 0), 3),
        }
        for c in chunks[:5]
    ]


def _build_messages(question: str, context: str) -> list:
    user_prompt = (
        f"CONTEXT FROM MEDICAL DOCUMENTS:\n{'=' * 50}\n{context}\n{'=' * 50}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer following the guidelines. Reply with the answer only."
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]


def _groq_kwargs(messages: list, stream: bool = False) -> dict:
    kwargs = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "stream": stream,
    }
    # Disable Qwen thinking. Do not set reasoning_format="hidden" — that still
    # spends time on a hidden chain-of-thought before the first visible token.
    if "qwen" in GROQ_MODEL.lower():
        kwargs["reasoning_effort"] = "none"
    return kwargs


def _create_completion(client: Groq, kwargs: dict):
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"⚠️ Groq create failed ({e}); retrying with compatibility params")
        fallback = {
            k: v
            for k, v in kwargs.items()
            if k not in ("reasoning_effort", "reasoning_format", "include_reasoning")
        }
        if "max_completion_tokens" in fallback:
            fallback["max_tokens"] = fallback.pop("max_completion_tokens")
        return client.chat.completions.create(**fallback)


class _IncrementalThinkStrip:
    """Yield only the visible (non-<think>) suffix as tokens arrive."""

    def __init__(self):
        self.raw = ""
        self.shown = ""

    def push(self, piece: str) -> str:
        if not piece:
            return ""
        self.raw += piece
        cleaned = strip_think_tags(self.raw)
        if cleaned.startswith(self.shown):
            delta = cleaned[len(self.shown):]
            self.shown = cleaned
            return delta
        self.shown = cleaned
        return ""


def generate_answer(question: str, chunks: list) -> tuple:
    """Generate answer using retrieved context with improved response quality."""
    if not chunks:
        return (
            "I don't have sufficient information in the medical documents to answer that question.",
            [],
        )

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        context = "\n\n".join([c["text"] for c in chunks[:5]])
        return (
            "I couldn't connect to the AI service due to a configuration issue. "
            "However, here's the relevant information from the documents:\n\n"
            f"{context}",
            [],
        )

    try:
        client = Groq(api_key=api_key)
        context = _format_context(chunks)
        messages = _build_messages(question, context)
        response = _create_completion(client, _groq_kwargs(messages, stream=False))
        answer = strip_think_tags(response.choices[0].message.content or "")
        if not answer:
            answer = "I couldn't produce a clear answer from the documents. Please try asking again."
        return answer, _sources_from_chunks(chunks)
    except Exception as e:
        print(f"⚠️ generate_answer failed: {e}")
        return (
            "I encountered an error while generating the response. "
            "Please try again with your question.",
            [],
        )


def iter_answer_tokens(question: str, chunks: list):
    """Yield visible answer text as Groq tokens arrive."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        context = "\n\n".join([c["text"] for c in chunks[:5]])
        yield (
            "I couldn't connect to the AI service due to a configuration issue. "
            "However, here's the relevant information from the documents:\n\n"
            f"{context}"
        )
        return

    client = Groq(api_key=api_key)
    messages = _build_messages(question, _format_context(chunks))
    stream = _create_completion(client, _groq_kwargs(messages, stream=True))
    stripper = _IncrementalThinkStrip()
    for event in stream:
        try:
            piece = event.choices[0].delta.content or ""
        except (AttributeError, IndexError):
            piece = ""
        visible = stripper.push(piece)
        if visible:
            yield visible
    if not stripper.shown:
        yield "I couldn't produce a clear answer from the documents. Please try asking again."


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _no_chunks_answer(channel: str) -> str:
    n = count_ready_chunks()
    docs = count_ready_documents()
    if channel == "whatsapp":
        if n is None:
            return "The knowledge base is not connected yet. Please try again later."
        if (docs or 0) == 0 or n == 0:
            return "I don't have any documents loaded yet. Please try again later."
        return (
            "I couldn't find relevant passages for that question. "
            "Try rephrasing it."
        )
    if n is None:
        return (
            "The knowledge base could not be opened. Check PINECONE_API_KEY on the "
            "API service, then try again."
        )
    if (docs or 0) == 0 or n == 0:
        return (
            "The knowledge base is empty. Add PDFs to backend/data and run "
            "POST /ingest (or: python main.py --ingest)."
        )
    return (
        "I couldn't find relevant passages for that question. "
        "Try rephrasing, or re-ingest with: python main.py --ingest --force"
    )


def run_chat(message: str, *, channel: str = "api") -> "ChatResponse":
    """Shared RAG pipeline used by the web /chat endpoint and WhatsApp."""
    text = (message or "").strip()
    if not text:
        return ChatResponse(
            answer="Please send a question about the medical documents.",
            sources=[],
            confidence=0.0,
        )

    if channel == "whatsapp" and _WA_GREETING.match(text):
        return ChatResponse(
            answer=(
                "Hi, I'm MedChat. Send a question and I'll answer from the "
                "medical documents."
            ),
            sources=[],
            confidence=1.0,
        )

    chunks = retrieve_chunks(text)
    confidence = sum(c["score"] for c in chunks) / len(chunks) if chunks else 0.0
    if not chunks:
        return ChatResponse(answer=_no_chunks_answer(channel), sources=[], confidence=0.0)

    answer, sources = generate_answer(text, chunks)
    return ChatResponse(answer=answer, sources=sources, confidence=confidence)


# ──────────────────────────────────────────────
# WhatsApp Cloud API
# ──────────────────────────────────────────────
def whatsapp_configured() -> bool:
    return bool(WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID)


def _already_processed_wa(msg_id: str) -> bool:
    if not msg_id:
        return False
    if msg_id in _PROCESSED_WA_IDS:
        return True
    _PROCESSED_WA_IDS.add(msg_id)
    while len(_PROCESSED_WA_IDS) > _PROCESSED_WA_MAX:
        _PROCESSED_WA_IDS.pop()
    return False


def _valid_whatsapp_signature(raw_body: bytes, header: str) -> bool:
    if not WHATSAPP_APP_SECRET:
        return True
    if not header.startswith("sha256="):
        return False
    expected = hmac.new(
        WHATSAPP_APP_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(header[7:], expected)


def format_whatsapp_reply(result: "ChatResponse") -> str:
    answer = re.sub(r"^#{1,3}\s+(.+)$", r"*\1*", (result.answer or "").strip(), flags=re.M)
    parts = [answer]
    if result.sources:
        lines = []
        seen = set()
        for src in result.sources:
            title = (src.get("title") or src.get("source") or "Document").strip()
            page = src.get("page")
            label = f"{title} p.{page}" if page else title
            if label in seen:
                continue
            seen.add(label)
            lines.append(f"• {label}")
        if lines:
            parts.append("Sources:\n" + "\n".join(lines[:5]))
    return "\n\n".join(p for p in parts if p)


def _split_whatsapp_text(text: str) -> list:
    if len(text) <= WHATSAPP_MAX_CHARS:
        return [text] if text else []
    parts = []
    remaining = text
    while remaining:
        if len(remaining) <= WHATSAPP_MAX_CHARS:
            parts.append(remaining)
            break
        window = remaining[:WHATSAPP_MAX_CHARS]
        cut = window.rfind("\n")
        if cut < WHATSAPP_MAX_CHARS // 2:
            cut = window.rfind(" ")
        if cut < WHATSAPP_MAX_CHARS // 2:
            cut = WHATSAPP_MAX_CHARS
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [p for p in parts if p]


def send_whatsapp_text(to: str, body: str) -> None:
    if not whatsapp_configured():
        print("⚠️ WhatsApp reply skipped: set WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID")
        return

    url = (
        f"https://graph.facebook.com/{WHATSAPP_GRAPH_VERSION}/"
        f"{WHATSAPP_PHONE_NUMBER_ID}/messages"
    )
    payload = json.dumps({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"preview_url": False, "body": body},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"⚠️ WhatsApp send failed ({e.code}): {detail}")
    except Exception as e:
        print(f"⚠️ WhatsApp send failed: {e}")


def handle_whatsapp_payload(payload: dict) -> None:
    if payload.get("object") != "whatsapp_business_account":
        return

    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            for msg in value.get("messages") or []:
                msg_id = msg.get("id") or ""
                if _already_processed_wa(msg_id):
                    continue

                sender = msg.get("from")
                if not sender:
                    continue

                msg_type = msg.get("type")
                if msg_type != "text":
                    send_whatsapp_text(
                        sender,
                        "Please send a text question and I'll answer from the medical documents.",
                    )
                    continue

                question = ((msg.get("text") or {}).get("body") or "").strip()
                if not question:
                    continue

                try:
                    result = run_chat(question, channel="whatsapp")
                    reply = format_whatsapp_reply(result)
                except Exception as e:
                    print(f"⚠️ WhatsApp chat failed: {e}")
                    reply = "I encountered an error while generating the response. Please try again."

                for chunk in _split_whatsapp_text(reply):
                    send_whatsapp_text(sender, chunk)


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "MedChat API", "docs": "/docs"}


@app.get("/health")
async def health():
    docs = count_ready_documents()
    chunks = count_ready_chunks()
    return {
        "status": "healthy",
        "store": "pinecone",
        "pinecone_index": store_uri(),
        "indexed_chunks": chunks or 0,
        "documents": docs,
        "chunks": chunks,
        "embedding_loaded": MODEL is not None,
        "embedding_backend": MODEL_BACKEND,
        "embedding_model": embed_model_name(),
        "embedding_dim": embedding_dim(),
        "pdf_files": [os.path.basename(p) for p in list_pdf_files()],
        "groq_model": GROQ_MODEL,
        "whatsapp_configured": whatsapp_configured() and bool(WHATSAPP_VERIFY_TOKEN),
    }


@app.get("/documents")
async def documents_list():
    try:
        return {"documents": list_documents()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def documents_delete(document_id: str):
    try:
        return delete_document(document_id)
    except DocumentNotFound:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(force: bool = False):
    try:
        return ingest_data_files(force=force)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return run_chat(req.message, channel="api")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE stream used by the web chatbot so tokens appear immediately."""
    text = (req.message or "").strip()

    async def events():
        yield _sse({"type": "status", "message": "searching"})
        if not text:
            yield _sse({
                "type": "token",
                "content": "Please send a question about the medical documents.",
            })
            yield _sse({"type": "done", "sources": [], "confidence": 0.0})
            return

        try:
            chunks = await asyncio.to_thread(retrieve_chunks, text)
        except Exception as e:
            print(f"⚠️ stream retrieve failed: {e}")
            yield _sse({
                "type": "error",
                "error": "I encountered an error while searching the documents.",
            })
            return

        if not chunks:
            yield _sse({"type": "token", "content": _no_chunks_answer("api")})
            yield _sse({"type": "done", "sources": [], "confidence": 0.0})
            return

        confidence = sum(c["score"] for c in chunks) / len(chunks)
        yield _sse({
            "type": "meta",
            "sources": _sources_from_chunks(chunks),
            "confidence": confidence,
        })

        try:
            for token in iter_answer_tokens(text, chunks):
                yield _sse({"type": "token", "content": token})
                await asyncio.sleep(0)
        except Exception as e:
            print(f"⚠️ stream generate failed: {e}")
            yield _sse({
                "type": "error",
                "error": "I encountered an error while generating the response.",
            })
            return

        yield _sse({"type": "done"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/whatsapp/webhook")
async def whatsapp_verify(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """Meta webhook verification handshake."""
    if (
        hub_mode == "subscribe"
        and WHATSAPP_VERIFY_TOKEN
        and hub_verify_token == WHATSAPP_VERIFY_TOKEN
    ):
        return PlainTextResponse(hub_challenge or "")
    raise HTTPException(status_code=403, detail="WhatsApp webhook verification failed")


@app.post("/whatsapp/webhook")
async def whatsapp_incoming(request: Request, background_tasks: BackgroundTasks):
    """Inbound WhatsApp messages → same RAG chatbot as POST /chat."""
    raw = await request.body()
    if WHATSAPP_APP_SECRET and not _valid_whatsapp_signature(
        raw, request.headers.get("X-Hub-Signature-256", "")
    ):
        raise HTTPException(status_code=403, detail="Invalid WhatsApp signature")

    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return {"status": "ignored"}

    background_tasks.add_task(handle_whatsapp_payload, payload)
    return {"status": "ok"}


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "rating": req.rating,
        "accuracy": req.accuracy,
        "categories": req.categories,
        "comments": req.comments,
    }

    feedbacks = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                feedbacks = json.load(f)
        except json.JSONDecodeError:
            pass

    feedbacks.append(entry)
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedbacks, f, indent=2)

    return {"success": True, "count": len(feedbacks)}


@app.get("/feedback/stats")
async def feedback_stats():
    if not os.path.exists(FEEDBACK_FILE):
        return {"total": 0, "avg_rating": 0, "avg_accuracy": 0}

    try:
        with open(FEEDBACK_FILE, "r") as f:
            feedbacks = json.load(f)
    except:
        return {"total": 0, "avg_rating": 0, "avg_accuracy": 0}

    if not feedbacks:
        return {"total": 0, "avg_rating": 0, "avg_accuracy": 0}

    avg_r = sum(f["rating"] for f in feedbacks) / len(feedbacks)
    avg_a = sum(f["accuracy"] for f in feedbacks) / len(feedbacks)

    return {
        "total": len(feedbacks),
        "avg_rating": round(avg_r, 1),
        "avg_accuracy": round(avg_a, 1),
    }


if __name__ == "__main__":
    import sys
    import uvicorn

    if "--ingest" in sys.argv:
        force = "--force" in sys.argv
        print(ingest_data_files(force=force))
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
