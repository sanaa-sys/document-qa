# 📄 Document question answering template

A simple Streamlit app that answers questions related to prophetic medicine. This is the structure of the code:
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────┐
│   PDF File  │───▶│  Text Chunking   │───▶│  HF API     │───▶│   Qdrant     │
│             │    │  (PyMuPDF)       │    │ (Embeddings)│    │ (Vector DB)  │
└─────────────┘    └──────────────────┘    └─────────────┘    └──────────────┘
                                                                      │
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐           │
│  Response   │◀───│    Groq API      │◀───│  Context +  │◀──────────┘
│             │    │    (LLM)         │    │   Query     │
└─────────────┘    └──────────────────┘    └─────────────┘

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://document-question-answering-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
