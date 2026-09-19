import hashlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
import json
import os
from datetime import datetime
import re

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

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
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_log.json")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DIM = 384
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 200
INGEST_BATCH_SIZE = 50
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.05"))
MATCH_COUNT = int(os.getenv("MATCH_COUNT", "5"))

INDEX = None
CHUNKS = None
MODEL = None
MODEL_BACKEND = None  # "fastembed" | "sentence-transformers"
SUPABASE = None


def _normalize_supabase_url(url: str) -> str:
    """Accept project URL or accidental /rest/v1 suffix."""
    cleaned = (url or "").strip().strip("'\"")
    cleaned = re.sub(r"/rest/v1/?$", "", cleaned)
    return cleaned.rstrip("/")


def get_supabase():
    global SUPABASE
    if SUPABASE is not None:
        return SUPABASE
        
    url = _normalize_supabase_url(os.getenv("SUPABASE_URL", ""))
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        return None

    from supabase import create_client

    # supabase-py appends /rest/v1 itself; passing that suffix causes PGRST125.
    SUPABASE = create_client(url, key)
    return SUPABASE


def ensure_model():
    """Lazy-load embeddings. Prefer fastembed (ONNX) so Render 512MB can boot."""
    global MODEL, MODEL_BACKEND
    if MODEL is not None:
        return MODEL

    try:
        from fastembed import TextEmbedding

        MODEL = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        MODEL_BACKEND = "fastembed"
        print(f"✅ Embedding backend: fastembed ({EMBEDDING_MODEL_NAME})")
        return MODEL
    except Exception as e:
        print(f"⚠️ fastembed unavailable ({e}); trying sentence-transformers")

    try:
        from sentence_transformers import SentenceTransformer

        MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        MODEL_BACKEND = "sentence-transformers"
        print(f"✅ Embedding backend: sentence-transformers ({EMBEDDING_MODEL_NAME})")
        return MODEL
    except Exception as e:
        raise RuntimeError(
            "No embedding backend installed. From the backend folder run: "
            "pip install fastembed"
        ) from e


def embed_texts(texts: list) -> list:
    """Return L2-normalized embedding vectors as plain Python float lists."""
    model = ensure_model()
    if MODEL_BACKEND == "fastembed":
        return [[float(x) for x in vec] for vec in model.embed(texts)]

    raw = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [[float(x) for x in vector] for vector in raw]


def file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def count_ready_chunks() -> Optional[int]:
    client = get_supabase()
    if client is None:
        return None
    try:
        res = (
            client.table("chunks")
            .select("id", count="exact")
            .limit(1)
            .execute()
        )
        return res.count if res.count is not None else 0
    except Exception as e:
        print(f"⚠️ chunks count failed: {e}")
        return None


def count_ready_documents() -> Optional[int]:
    client = get_supabase()
    if client is None:
        return None
    try:
        res = (
            client.table("documents")
            .select("id", count="exact")
            .eq("status", "ready")
            .limit(1)
            .execute()
        )
        return res.count if res.count is not None else 0
    except Exception as e:
        print(f"⚠️ documents count failed: {e}")
        return None


def _rows_from_match(data) -> list:
    results = []
    for row in data or []:
        results.append({
            "text": row.get("text", ""),
            "page": row.get("page", 0),
            "source": row.get("source") or row.get("title") or "Document",
            "title": row.get("title") or row.get("source") or "Document",
            "document_id": row.get("document_id"),
            "score": float(row.get("score") or 0.0),
        })
    return results


def retrieve_chunks_from_supabase(question: str, top_k: int = MATCH_COUNT) -> list:
    client = get_supabase()
    if client is None:
        print("⚠️ Supabase client missing during retrieval")
        return []

    try:
        question_vec = embed_texts([question])[0]
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return []

    if len(question_vec) != EMBEDDING_DIM:
        print(f"⚠️ Unexpected embedding dim {len(question_vec)} (expected {EMBEDDING_DIM})")

    for threshold in (MATCH_THRESHOLD, 0.0):
        try:
            res = client.rpc(
                "match_chunks",
                {
                    "query_embedding": question_vec,
                    "match_threshold": threshold,
                    "match_count": top_k,
                    "filter_document_id": None,
                },
            ).execute()
            hits = _rows_from_match(res.data)
            if hits:
                print(f"✅ Vector hits: {len(hits)} (threshold={threshold}, best={hits[0]['score']:.3f})")
                return hits
        except Exception as e:
            print(f"⚠️ match_chunks failed (threshold={threshold}): {e}")

    # Keyword / full-text style fallback via ilike on chunks + join-like title from documents list
    try:
        tokens = re.findall(r"[A-Za-z0-9]{3,}", question.lower())[:5]
        query = client.table("chunks").select("page,text,document_id,documents(title,filename)")
        if tokens:
            or_filter = ",".join([f"text.ilike.%{tok}%" for tok in tokens])
            query = query.or_(or_filter)
        res = query.limit(top_k).execute()
        hits = []
        for row in res.data or []:
            doc = row.get("documents") or {}
            hits.append({
                "text": row.get("text", ""),
                "page": row.get("page", 0),
                "source": doc.get("filename") or doc.get("title") or "Document",
                "title": doc.get("title") or doc.get("filename") or "Document",
                "document_id": row.get("document_id"),
                "score": 0.0,
            })
        if hits:
            print(f"✅ Keyword fallback hits: {len(hits)}")
            return hits
        print("⚠️ No chunk hits — is the documents/chunks schema empty? Run /ingest.")
    except Exception as e:
        print(f"⚠️ Keyword fallback failed: {e}")

    return []


def retrieve_chunks(question: str, top_k: int = MATCH_COUNT) -> list:
    supabase_hits = retrieve_chunks_from_supabase(question, top_k=top_k)
    if supabase_hits:
        return supabase_hits

    if faiss is None or INDEX is None or CHUNKS is None:
        return []

    try:
        ensure_model()
    except Exception:
        return []

    import numpy as np

    question_vec = np.array(embed_texts([question]), dtype="float32")
    if MODEL_BACKEND != "fastembed" and faiss is not None:
        faiss.normalize_L2(question_vec)

    distances, indices = INDEX.search(question_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(CHUNKS) and dist > MATCH_THRESHOLD:
            results.append({
                "text": CHUNKS[idx]["text"],
                "page": CHUNKS[idx]["page"],
                "source": CHUNKS[idx]["source"],
                "title": CHUNKS[idx].get("source", "Document"),
                "score": float(dist),
            })
    return results


@app.on_event("startup")
async def load_rag():
    """Boot fast: do not load embedding models or FAISS unless explicitly asked."""
    global INDEX, CHUNKS

    docs = count_ready_documents()
    chunks = count_ready_chunks()
    if docs is None:
        print("⚠️ Supabase not configured (set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)")
    else:
        print(f"✅ Supabase connected: {docs} ready documents, {chunks} chunks")

    load_faiss = os.getenv("LOAD_FAISS_ON_STARTUP", "").strip().lower() in ("1", "true", "yes")
    if load_faiss:
        try:
            index_path = os.path.join(DATA_DIR, "faiss_index.bin")
            chunks_path = os.path.join(DATA_DIR, "chunks_data.json")
            if faiss is not None and os.path.exists(index_path) and os.path.exists(chunks_path):
                INDEX = faiss.read_index(index_path)
                with open(chunks_path, "r") as f:
                    CHUNKS = json.load(f)
                print(f"✅ FAISS loaded: {len(CHUNKS)} chunks indexed")
            elif faiss is None:
                print("⚠️ FAISS not installed; using Supabase retrieval only")
        except Exception as e:
            print(f"⚠️ FAISS load failed: {e}")

    if os.getenv("LOAD_EMBEDDING_ON_STARTUP", "").strip().lower() in ("1", "true", "yes"):
        try:
            ensure_model()
        except Exception as e:
            print(f"⚠️ Embedding model load failed: {e}")
    else:
        print("ℹ️ Embeddings will load lazily on first /chat (saves RAM at boot)")

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


def get_document_by_filename(client, filename: str) -> Optional[dict]:
    res = client.table("documents").select("*").eq("filename", filename).limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None


def upsert_chunk_batch(client, rows: list):
    for i in range(0, len(rows), INGEST_BATCH_SIZE):
        batch = rows[i:i + INGEST_BATCH_SIZE]
        client.table("chunks").upsert(
            batch,
            on_conflict="document_id,chunk_index",
        ).execute()


def list_documents() -> list:
    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase is not configured.")
    res = (
        client.table("documents")
        .select("id,title,filename,checksum,status,page_count,chunk_count,embedding_model,error,created_at,updated_at")
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


def delete_document(document_id: str) -> dict:
    client = get_supabase()
    if client is None:
        raise RuntimeError("Supabase is not configured.")
    existing = client.table("documents").select("id,filename").eq("id", document_id).limit(1).execute()
    if not existing.data:
        raise HTTPException(status_code=404, detail="Document not found")
    client.table("documents").delete().eq("id", document_id).execute()
    return {"deleted": True, "id": document_id, "filename": existing.data[0].get("filename")}


def ingest_one_pdf(client, path: str, force: bool = False) -> dict:
    filename = os.path.basename(path)
    title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip() or filename
    checksum = file_checksum(path)
    existing = get_document_by_filename(client, filename)

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
        client.table("documents").update({
            "title": title,
            "checksum": checksum,
            "status": "processing",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "error": None,
        }).eq("id", document_id).execute()
        # Replace chunks for this document
        client.table("chunks").delete().eq("document_id", document_id).execute()
    else:
        inserted = client.table("documents").insert({
            "title": title,
            "filename": filename,
            "checksum": checksum,
            "status": "processing",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "metadata": {"path": filename},
        }).execute()
        document_id = inserted.data[0]["id"]

    try:
        page_count, rows = extract_pdf_chunks(path)
        if not rows:
            client.table("documents").update({
                "status": "failed",
                "error": "No extractable text",
                "page_count": page_count,
                "chunk_count": 0,
            }).eq("id", document_id).execute()
            return {
                "filename": filename,
                "status": "failed",
                "reason": "no extractable text",
                "document_id": document_id,
                "chunks_upserted": 0,
            }

        embeddings = embed_texts([row["text"] for row in rows])
        chunk_rows = []
        for row, vector in zip(rows, embeddings):
            chunk_rows.append({
                "document_id": document_id,
                "chunk_index": row["chunk_index"],
                "page": row["page"],
                "text": row["text"],
                "embedding": vector,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "metadata": row.get("metadata") or {},
            })

        upsert_chunk_batch(client, chunk_rows)
        client.table("documents").update({
            "status": "ready",
            "page_count": page_count,
            "chunk_count": len(chunk_rows),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "error": None,
        }).eq("id", document_id).execute()

        print(f"✅ Ingested {filename}: {len(chunk_rows)} chunks ({page_count} pages)")
        return {
            "filename": filename,
            "status": "ingested",
            "document_id": document_id,
            "chunks_upserted": len(chunk_rows),
            "page_count": page_count,
        }
    except Exception as e:
        client.table("documents").update({
            "status": "failed",
            "error": str(e)[:500],
        }).eq("id", document_id).execute()
        raise


def ingest_data_files(force: bool = False) -> dict:
    """Parse PDFs in ./data into documents + chunks."""
    client = get_supabase()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY, "
            "and run backend/supabase_schema.sql in the Supabase SQL editor."
        )

    ensure_model()

    pdfs = list_pdf_files()
    if not pdfs:
        return {
            "ingested_files": [],
            "skipped_files": [],
            "failed_files": [],
            "chunks_upserted": 0,
            "documents": count_ready_documents(),
            "chunks": count_ready_chunks(),
        }

    ingested = []
    skipped = []
    failed = []
    chunks_upserted = 0

    for path in pdfs:
        try:
            result = ingest_one_pdf(client, path, force=force)
            if result["status"] == "skipped":
                skipped.append(result["filename"])
            elif result["status"] == "failed":
                failed.append(result["filename"])
            else:
                ingested.append(result["filename"])
                chunks_upserted += result.get("chunks_upserted", 0)
        except Exception as e:
            failed.append(os.path.basename(path))
            print(f"⚠️ Ingest failed for {path}: {e}")

    return {
        "ingested_files": ingested,
        "skipped_files": skipped,
        "failed_files": failed,
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


def generate_answer(question: str, chunks: list) -> tuple:
    """Generate answer using retrieved context with improved response quality."""
    
    if not chunks:
        return "I don't have sufficient information in the medical documents to answer that question.", []
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        context = "\n\n".join([c["text"] for c in chunks[:5]])
        return (
            "I couldn't connect to the AI service due to a configuration issue. "
            "However, here's the relevant information from the documents:\n\n"
            f"{context}", 
            []
        )
    
    try:
        client = Groq(api_key=api_key)
        
        # Format context with clear separators and page references
        context_parts = [
            f"[{c.get('title') or c.get('source') or 'Document'} · p.{c['page']}]: {c['text']}"
            for c in chunks[:5]
        ]
        context = "\n\n---\n\n".join(context_parts)
        
        # Improved system prompt for better response quality
        system_instruction = (
            "You are a medical knowledge assistant providing accurate, well-formatted responses. "
            "Follow these guidelines strictly:\n\n"
            "1. Answer ONLY using the provided context from medical documents.\n"
            "2. If the information isn't in the context, state clearly: 'I don't have sufficient "
            "information in the medical documents to answer that question.'\n"
            "3. Write in clear, professional language with complete sentences.\n"
            "4. Use bullet points or numbered lists when presenting multiple pieces of information.\n"
            "5. Cite page numbers when referencing specific information (e.g., 'According to page 49...').\n"
            "6. Avoid speculation, generalizations, or adding information outside the context.\n"
            "7. If context is contradictory, acknowledge the discrepancy.\n\n"
            "Format your response with:\n"
            "- A direct answer to the question in the first paragraph\n"
            "- Supporting details organized logically\n"
            "- Clear attribution to source pages\n\n"
            "Never include <think> tags, chain-of-thought, or internal reasoning in the reply."
        )
        
        user_prompt = (
            f"CONTEXT FROM MEDICAL DOCUMENTS:\n{'='*50}\n{context}\n{'='*50}\n\n"
            f"QUESTION: {question}\n\n"
            "Provide a well-structured, professional answer following all guidelines above."
        )
        
        create_kwargs = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": 1024,
            "stream": False,
        }
        # Qwen reasoning models otherwise dump <think> into the user-visible answer.
        if "qwen" in GROQ_MODEL.lower():
            create_kwargs["reasoning_format"] = "hidden"
            create_kwargs["reasoning_effort"] = "none"

        try:
            response = client.chat.completions.create(**create_kwargs)
        except Exception:
            create_kwargs.pop("reasoning_format", None)
            create_kwargs.pop("reasoning_effort", None)
            response = client.chat.completions.create(**create_kwargs)
        
        answer = strip_think_tags(response.choices[0].message.content or "")
        if not answer:
            answer = "I couldn't produce a clear answer from the documents. Please try asking again."
        
        # Clean sources to include only essential metadata
        sources = [
            {
                "page": c["page"],
                "source": os.path.basename(str(c.get("source") or "Document")),
                "title": c.get("title") or c.get("source") or "Document",
                "document_id": c.get("document_id"),
                "score": round(float(c.get("score") or 0), 3),
            }
            for c in chunks[:5]
        ]
        
        return answer, sources
    
    except Exception as e:
        error_detail = str(e)
        # Don't expose internal error details to users
        return (
            "I encountered an error while generating the response. "
            "Please try again with your question.",
            []
        )


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
    faiss_chunks = len(CHUNKS) if CHUNKS else 0
    indexed_chunks = chunks if chunks is not None else faiss_chunks
    return {
        "status": "healthy",
        "indexed_chunks": indexed_chunks,
        "documents": docs,
        "chunks": chunks,
        "faiss_chunks": faiss_chunks,
        "supabase_configured": get_supabase() is not None,
        "embedding_loaded": MODEL is not None,
        "embedding_backend": MODEL_BACKEND,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "pdf_files": [os.path.basename(p) for p in list_pdf_files()],
        "groq_model": GROQ_MODEL,
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
    except HTTPException:
        raise
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
    chunks = retrieve_chunks(req.message)
    confidence = sum(c["score"] for c in chunks) / len(chunks) if chunks else 0.0
    if not chunks:
        n = count_ready_chunks()
        docs = count_ready_documents()
        if n is None:
            answer = (
                "The knowledge base is not connected. Set SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY on the API service, then try again."
            )
        elif (docs or 0) == 0 or n == 0:
            answer = (
                "The knowledge base is empty. Add PDFs to backend/data and run "
                "POST /ingest (or: python main.py --ingest)."
            )
        else:
            answer = (
                "I couldn't find relevant passages for that question. "
                "Try rephrasing, or re-ingest with: python main.py --ingest --force"
            )
        return ChatResponse(answer=answer, sources=[], confidence=0.0)

    answer, sources = generate_answer(req.message, chunks)
    return ChatResponse(answer=answer, sources=sources, confidence=confidence)


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
