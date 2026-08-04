import streamlit as st
from streamlit_option_menu import option_menu
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import requests
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
@st.cache_resource
def load_rag_components():
    """Load index + model ONCE (cached for entire app lifetime)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # Load FAISS index
    index = faiss.read_index(os.path.join(data_dir, "faiss_index.bin"))

    # Load chunks
    with open(os.path.join(data_dir, "chunks_data.json"), "r") as f:
        chunks = json.load(f)

    # Load embedding model (lightweight, ~80 MB)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    return index, chunks, model


def retrieve_relevant_chunks(question, index, chunks, model, top_k=5):
    """Search the FAISS index for the most relevant chunks"""
    # Embed the question
    question_vec = model.encode([question])
    faiss.normalize_L2(question_vec)

    # Search
    distances, indices = index.search(question_vec.astype('float32'), top_k)

    # Gather results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks) and dist > 0.3:  # relevance threshold
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "source": chunks[idx]["source"],
                "score": float(dist)
            })
    return results


def generate_answer(question, retrieved_chunks):
    """Call your LLM API to generate a grounded answer"""
    api_key = st.secrets["GROQ_API_KEY"]

    if not api_key:
        # Fallback: return raw chunks if no API key configured
        context = "\n\n".join([c["text"] for c in retrieved_chunks])
        return (
            f"I found {len(retrieved_chunks)} relevant passages, but no LLM API key "
            f"is configured. Here's the most relevant context:\n\n{context}"
        )

    context = "\n\n---\n\n".join([
        f"[Page {c['page']}] {c['text']}" for c in retrieved_chunks
    ])

    prompt = f"""You are a medical knowledge assistant. Answer the question based 
ONLY on the context below. If the context doesn't contain the answer, say 
"I don't have sufficient information to answer that."

Context:
{context}

Question: {question}
"""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 500
            },
            timeout=30
        )
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating answer: {str(e)}"


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="MedChat", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main-title {
        background-color: #6d4aff;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .main-title h1 { color: white; font-size: 1.5rem; margin: 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"><h1>🩺 MedChat — Medical Knowledge Assistant</h1></div>',
            unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["About", "Chatbot", "Feedback"],
    icons=["info-circle", "chat-dots-fill", "envelope-heart"],
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f0f0", "border-radius": "10px"},
        "nav-link": {"font-size": "1rem", "font-weight": "500", "text-align": "center", "border-radius": "10px"},
        "nav-link-selected": {"background-color": "#6d4aff", "color": "white", "font-weight": "600", "border-radius": "10px"},
    },
)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_list" not in st.session_state:
    st.session_state.feedback_list = []
if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False

# ──────────────────────────────────────────────
# LOAD RAG COMPONENTS (cached, runs once)
# ──────────────────────────────────────────────
try:
    index, chunks, model = load_rag_components()
    st.session_state.rag_loaded = True
except Exception as e:
    st.session_state.rag_loaded = False
    load_error = str(e)


# ═════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════
if selected == "About":
    st.markdown("## 📖 About MedChat")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### What is MedChat?
        MedChat is a **retrieval-augmented medical knowledge assistant** that answers 
        questions based on a pre-indexed medical reference document.

        ### How It Works
        1. A large medical PDF was processed into searchable text chunks
        2. Each chunk was converted to a vector embedding
        3. When you ask a question, the system finds the most relevant passages
        4. An AI model generates an answer grounded ONLY in those passages

        ### Features
        - 🔒 Answers grounded only in the source document
        - 📄 Source citations with page numbers
        - 💬 Conversational interface with chat history
        - ⚡ Fast vector search (FAISS)
        """)
    with col2:
        st.markdown("### ⚠️ Disclaimer")
        st.warning(
            "MedChat is **informational only** and NOT a substitute for "
            "professional medical advice. Always consult a healthcare provider."
        )
        st.markdown("### 📊 System Info")
        if st.session_state.rag_loaded:
            st.metric("Indexed chunks", f"{len(chunks):,}")
            st.metric("Embedding dim", f"{index.d}")
        else:
            st.error("Index not loaded!")

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    cols = st.columns(4)
    for col, (name, role) in zip(cols, [
        ("Streamlit", "Web UI"), ("FAISS", "Vector Search"),
        ("SentenceTransformers", "Embeddings"), ("GPT-4o-mini", "LLM")
    ]):
        col.metric(name, role)


# ═════════════════════════════════════════════
# PAGE: CHATBOT
# ═════════════════════════════════════════════
elif selected == "Chatbot":
    st.markdown("### 💬 Medical Chatbot")
    st.caption("Ask questions about the indexed medical reference document.")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Options")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        if st.session_state.rag_loaded:
            st.success(f"📚 {len(chunks):,} chunks indexed")
        else:
            st.error("Index failed to load")
        st.markdown("---")
        st.markdown("**Tip:** Ask specific questions like:")
        st.caption("• What are the dosage guidelines for amoxicillin?")
        st.caption("• What are the symptoms of acute pancreatitis?")
        st.caption("• Explain the protocol for sepsis management")

    # Chat display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(f"<small>📎 Sources: {msg['sources']}</small>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask a medical question...")

    if user_input:
        if not st.session_state.rag_loaded:
            st.error("⚠️ The knowledge base is not loaded. Please contact support.")
            st.stop()

        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching medical document..."):
                # Step 1: Retrieve relevant chunks
                retrieved = retrieve_relevant_chunks(
                    user_input, index, chunks, model, top_k=5
                )

                if not retrieved:
                    response = "I don't have sufficient information in the indexed document to answer that question."
                    sources = None
                else:
                    # Step 2: Generate answer with LLM
                    response = generate_answer(user_input, retrieved)
                    sources = "; ".join(
                        [f"{r['source']} — p.{r['page']}" for r in retrieved]
                    )

                st.markdown(response)
                if sources:
                    st.markdown(f"<small>📎 Sources: {sources}</small>", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources if retrieved else None,
        })


# ═════════════════════════════════════════════
# PAGE: FEEDBACK
# ═════════════════════════════════════════════
elif selected == "Feedback":
    st.markdown("## 📝 Feedback Form")
    st.markdown("Help us improve MedChat by sharing your experience.")
    st.markdown("---")

    with st.form("feedback_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)

        with col_a:
            name = st.text_input("Name (optional)", placeholder="Dr. Jane Doe")
            role = st.selectbox("Your Role", [
                "Select...", "Doctor", "Nurse", "Researcher", "Student", "Patient", "Other"
            ])

        with col_b:
            rating = st.slider("Overall Rating", 1, 5, 4, help="1=Poor, 5=Excellent")
            accuracy = st.slider("Answer Accuracy", 1, 5, 4)

        feedback_type = st.multiselect("Feedback Category", [
            "Answer Quality", "Speed", "UI/UX", "Missing Feature", "Bug Report", "Other"
        ])

        comments = st.text_area(
            "Detailed Comments",
            placeholder="What went well? What could be improved?",
            height=150,
        )

        submitted = st.form_submit_button("Submit Feedback", type="primary")

        if submitted:
            if not comments.strip():
                st.warning("Please add some comments.")
            else:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "name": name or "Anonymous",
                    "role": role,
                    "rating": rating,
                    "accuracy": accuracy,
                    "categories": feedback_type,
                    "comments": comments,
                }
                st.session_state.feedback_list.append(entry)
                st.success("✅ Thank you! Your feedback has been recorded.")
                st.balloons()

    # Show recent feedback
    if st.session_state.feedback_list:
        st.markdown("---")
        st.markdown("### 📊 Recent Feedback")
        for fb in reversed(st.session_state.feedback_list[-5:]):
            with st.expander(f"{fb['name']} — {'⭐'*fb['rating']} — {fb['timestamp'][:10]}"):
                st.write(f"**Role:** {fb['role']}")
                st.write(f"**Categories:** {', '.join(fb['categories']) or 'None'}")
                st.write(f"**Comments:** {fb['comments']}")
