import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- CONFIGURATION ---
# Set your keys here or via environment variables
GROQ_API_KEY = api_key=st.secrets["GROQ_API_KEY"]
PDF_PATH = "Medicine.pdf"

st.set_page_config(page_title="Medical AI Assistant", layout="centered")

# --- UI Header ---
st.title("🩺 Medicine PDF Assistant")
st.markdown(f"This chatbot is trained on the English Translation of The Prophetic Medicine written by Ibn Qayyim Al Jawziya. Ask any questions about its content below.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Ingestion Logic (Runs once on startup) ---
def initialize_knowledge_base():
    if not os.path.exists(PDF_PATH):
        st.error(f"Error: {PDF_PATH} not found in the directory.")
        return None
    
    with st.spinner("Initializing medical knowledge base..."):
        # 1. Load the specific PDF
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        
        # 2. Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        # 3. Create Embeddings & Store in-memory
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            location=":memory:", # Fast, no database setup needed
            collection_name="medicine_knowledge"
        )
        return vector_store

# Run initialization once
if st.session_state.vector_store is None:
    st.session_state.vector_store = initialize_knowledge_base()

# --- Chat Interface ---

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question about the Prophetic Medicine document..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval and Generation
    with st.chat_message("assistant"):
        try:
            # 1. Retrieve relevant chunks
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            context_docs = retriever.invoke(prompt)
            context_text = "\n\n".join([doc.page_content for doc in context_docs])

            # 2. Call Groq
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY, 
                model_name="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            rag_prompt = f""" You are an expert assistant specializing in Islamic Medicine (Tibb), 
            including Prophetic Medicine (Tibb-an-Nabawi) and Unani traditions.
            
            Your goal is to answer the user's question based STRICTLY on the provided context 
            from the "Prophetic Medicine.pdf" document.

            ### GUIDELINES FOR YOUR RESPONSE:
            1. **Accuracy:** Use only the information found in the context. If the context does not 
               mention a specific remedy or concept, state clearly: "The provided medical text 
               does not contain information regarding this."
            2. **Terminology:** Use traditional terms where appropriate (e.g., Mizaj for temperament, 
               Akhlat for humors) if they appear in the text.
            3. **Contextual Integrity:** If the document mentions spiritual aspects alongside 
               physical remedies (diet, herbs, or hygiene), include them to provide a holistic 
               view as per the Islamic medical tradition.
            4. **Safety Disclaimer:** Always include a brief note that traditional remedies 
               should be discussed with a qualified practitioner.
            5. **Formatting:** Use bullet points for remedies or lists of benefits to make 
               the information easy to read.
            
            Context:
            {context_text}
            
            Question: {prompt}
            """
            
            response = llm.invoke(rag_prompt)
            st.markdown(response.content)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
