from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import os
from datetime import datetime
import re


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="MedChat API")

# ──────────────────────────────────────────────
# CORS — allow Next.js frontend
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",       "https://1-3rd.vercel.app",
        "https://1-3rd.vercel.app/*","*"],  # Add production URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
# llama3-8b-8192 was decommissioned; override with GROQ_MODEL if needed.
GROQ_MODEL =  "qwen/qwen3.8-27b"
TEMPERATURE = 0.1
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_log.json")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 200
INGEST_BATCH_SIZE = 50
MATCH_THRESHOLD = 0.3

# ──────────────────────────────────────────────
# LOAD RAG COMPONENTS AT STARTUP
# ──────────────────────────────────────────────
INDEX = None
CHUNKS = None
MODEL = None
SUPABASE = None


def get_supabase():
    global SUPABASE
    if SUPABASE is not None:
        return SUPABASE

    url = (os.getenv("SUPABASE_URL") or "").strip().strip("'\"")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or ""
    ).strip().strip("'\"")
    if not url or not key:
        return None

    from supabase import create_client

    SUPABASE = create_client(url, key)
    return SUPABASE


def ensure_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return MODEL


def count_supabase_chunks() -> Optional[int]:
    client = get_supabase()
    if client is None:
        return None
    try:
        res = client.table("document_chunks").select("id", count="exact").limit(1).execute()
        return res.count if res.count is not None else 0
    except Exception as e:
        print(f"⚠️ Supabase count failed: {e}")
        return None


@app.on_event("startup")
async def load_rag():
    global INDEX, CHUNKS, MODEL
    try:
        MODEL = ensure_model()
        print(f"✅ Embedding model loaded: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"⚠️ Embedding model load failed: {e}")

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

    supabase_count = count_supabase_chunks()
    if supabase_count is None:
        print("⚠️ Supabase not configured (set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)")
    else:
        print(f"✅ Supabase connected: {supabase_count} chunks")

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
# PDF INGEST → SUPABASE
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


def extract_pdf_chunks(path: str) -> list:
    from pypdf import PdfReader

    reader = PdfReader(path)
    source = os.path.basename(path)
    rows = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        for chunk_index, text in enumerate(chunk_text(page_text)):
            rows.append({
                "source": source,
                "page": page_number,
                "chunk_index": chunk_index,
                "text": text,
            })
    return rows


def already_ingested_sources(client) -> set:
    try:
        res = client.table("document_chunks").select("source").execute()
        return {row["source"] for row in (res.data or [])}
    except Exception as e:
        print(f"⚠️ Could not list ingested sources: {e}")
        return set()


def upsert_chunk_batch(client, rows: list):
    for i in range(0, len(rows), INGEST_BATCH_SIZE):
        batch = rows[i:i + INGEST_BATCH_SIZE]
        client.table("document_chunks").upsert(
            batch,
            on_conflict="source,page,chunk_index",
        ).execute()


def ingest_data_files(force: bool = False) -> dict:
    """Parse PDFs in ./data, embed chunks, and upsert them into Supabase."""
    client = get_supabase()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY, "
            "and run backend/supabase_schema.sql in the Supabase SQL editor."
        )

    model = ensure_model()
    pdfs = list_pdf_files()
    if not pdfs:
        return {"ingested_files": [], "skipped_files": [], "chunks_upserted": 0}

    existing = set() if force else already_ingested_sources(client)
    ingested = []
    skipped = []
    chunks_upserted = 0

    for path in pdfs:
        source = os.path.basename(path)
        if source in existing:
            skipped.append(source)
            continue

        if force:
            try:
                client.table("document_chunks").delete().eq("source", source).execute()
            except Exception as e:
                print(f"⚠️ Could not delete existing rows for {source}: {e}")

        rows = extract_pdf_chunks(path)
        if not rows:
            skipped.append(source)
            print(f"⚠️ No extractable text in {source}")
            continue

        embeddings = model.encode(
            [row["text"] for row in rows],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for row, vector in zip(rows, embeddings):
            row["embedding"] = vector.tolist()

        upsert_chunk_batch(client, rows)
        ingested.append(source)
        chunks_upserted += len(rows)
        print(f"✅ Ingested {source}: {len(rows)} chunks")

    return {
        "ingested_files": ingested,
        "skipped_files": skipped,
        "chunks_upserted": chunks_upserted,
        "supabase_chunks": count_supabase_chunks(),
    }


def retrieve_chunks_from_supabase(question: str, top_k: int = 5) -> list:
    client = get_supabase()
    if client is None or MODEL is None:
        return []

    question_vec = MODEL.encode([question], normalize_embeddings=True)[0].tolist()
    try:
        res = client.rpc(
            "match_document_chunks",
            {
                "query_embedding": question_vec,
                "match_threshold": MATCH_THRESHOLD,
                "match_count": top_k,
            },
        ).execute()
    except Exception as e:
        print(f"⚠️ Supabase vector search failed: {e}")
        return []

    results = []
    for row in res.data or []:
        results.append({
            "text": row.get("text", ""),
            "page": row.get("page", 0),
            "source": row.get("source") or "Document",
            "score": float(row.get("score") or 0.0),
        })
    return results


# ──────────────────────────────────────────────
# RAG HELPERS
# ──────────────────────────────────────────────
def retrieve_chunks(question: str, top_k: int = 5) -> list:
    supabase_hits = retrieve_chunks_from_supabase(question, top_k=top_k)
    if supabase_hits:
        return supabase_hits

    if faiss is None or INDEX is None or CHUNKS is None or MODEL is None:
        return []

    question_vec = MODEL.encode([question])
    faiss.normalize_L2(question_vec)
    distances, indices = INDEX.search(question_vec.astype("float32"), top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(CHUNKS) and dist > MATCH_THRESHOLD:
            results.append({
                "text": CHUNKS[idx]["text"],
                "page": CHUNKS[idx]["page"],
                "source": CHUNKS[idx]["source"],
                "score": float(dist),
            })
    return results


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
            f"Source [Page {c['page']}]: {c['text']}"
            for c in chunks[:5]  # Limit to top 5 most relevant
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
                "source": os.path.basename(c["source"]) if c["source"] else "Document",
                "score": round(c["score"], 3)
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
    supabase_chunks = count_supabase_chunks()
    faiss_chunks = len(CHUNKS) if CHUNKS else 0
    indexed_chunks = supabase_chunks if supabase_chunks is not None else faiss_chunks
    return {
        "status": "healthy",
        "indexed_chunks": indexed_chunks,
        "faiss_chunks": faiss_chunks,
        "supabase_configured": get_supabase() is not None,
        "supabase_chunks": supabase_chunks,
        "pdf_files": [os.path.basename(p) for p in list_pdf_files()],
        "groq_model": GROQ_MODEL,
    }


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
