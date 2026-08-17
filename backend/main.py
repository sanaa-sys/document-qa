from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import os
from datetime import datetime

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
GROQ_MODEL =  "qwen/qwen3.6-27b"
TEMPERATURE = 0.1
FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "data", "feedback_log.json")

# ──────────────────────────────────────────────
# LOAD RAG COMPONENTS AT STARTUP
# ──────────────────────────────────────────────
INDEX = None
CHUNKS = None
MODEL = None

@app.on_event("startup")
async def load_rag():
    global INDEX, CHUNKS, MODEL
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    try:
        INDEX = faiss.read_index(os.path.join(data_dir, "faiss_index.bin"))
        with open(os.path.join(data_dir, "chunks_data.json"), "r") as f:
            CHUNKS = json.load(f)
        MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print(f"✅ RAG loaded: {len(CHUNKS)} chunks indexed")
    except Exception as e:
        print(f"⚠️ RAG load failed: {e}")


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
# RAG HELPERS
# ──────────────────────────────────────────────
def retrieve_chunks(question: str, top_k: int = 5) -> list:
    if INDEX is None or CHUNKS is None or MODEL is None:
        return []

    question_vec = MODEL.encode([question])
    faiss.normalize_L2(question_vec)
    distances, indices = INDEX.search(question_vec.astype("float32"), top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(CHUNKS) and dist > 0.3:
            results.append({
                "text": CHUNKS[idx]["text"],
                "page": CHUNKS[idx]["page"],
                "source": CHUNKS[idx]["source"],
                "score": float(dist),
            })
    return results


def generate_answer(question: str, chunks: list) -> tuple:
    if not chunks:
        return "I don't have sufficient information in the medical documents to answer that question.", []

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        context = "\n\n".join([c["text"] for c in chunks])
        return f"No Groq API key found. Raw passages:\n\n{context}", []

    try:
        client = Groq(api_key=api_key)
        context = "\n\n---\n\n".join(
            [f"[Page {c['page']}] {c['text']}" for c in chunks]
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical knowledge assistant. "
                        "Answer using ONLY the provided context from medical documents. "
                        "If the information isn't in the context, say you don't have "
                        "sufficient information to answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=500,
        )

        answer = response.choices[0].message.content
        sources = [
            {"page": c["page"], "source": c["source"], "score": c["score"]}
            for c in chunks
        ]
        return answer, sources
    except Exception as e:
        return f"Error generating answer: {str(e)}", []


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "MedChat API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "indexed_chunks": len(CHUNKS) if CHUNKS else 0,
        "groq_model": GROQ_MODEL,
    }


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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)