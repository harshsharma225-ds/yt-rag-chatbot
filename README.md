# 🎥 YouTube RAG Chatbot

Paste a YouTube video link, and ask questions about its content — powered by a Retrieval-Augmented Generation (RAG) pipeline built with LangChain.

## How it works

1. **Transcript extraction** — Pulls the video's transcript directly from YouTube (supports English and Hindi, with automatic fallback).
2. **Chunking** — Splits the transcript into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`, preserving context across boundaries.
3. **Embeddings + Vector Store** — Each chunk is embedded using a local Hugging Face sentence-transformer model and stored in a persistent Chroma vector database (keyed per video, so a video is never re-processed twice).
4. **Retrieval + Generation** — On each question, the most relevant chunks are retrieved and passed to Groq's LLM (`openai/gpt-oss-20b`) to generate a grounded answer.
5. **UI** — A Streamlit chat interface to load a video and ask questions interactively.

## Tech Stack

- **LangChain** — orchestration (chunking, retrieval chain, prompt handling)
- **youtube-transcript-api** — transcript extraction
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, no API cost
- **ChromaDB** — vector store, persisted locally per video
- **Groq API** (`openai/gpt-oss-20b`) — fast, free-tier LLM inference
- **Streamlit** — chat UI

## Project Structure

| Path | Description |
|------|-------------|
| `app.py` | Streamlit entry point |
| `src/transcript_loader.py` | Fetch + clean YouTube transcripts (EN/HI) |
| `src/chunking.py` | Split transcript into overlapping chunks |
| `src/vectorstore.py` | Embed + persist/load Chroma vectorstore per video |
| `src/qa_chain.py` | Retrieval + Groq LLM answer chain |
| `data/` | Cached transcripts (git-ignored) |
| `vectorstore/` | Persisted Chroma DBs per video (git-ignored) |
| `requirements.txt` | Python dependencies |
| `.env` | API keys (git-ignored) |
| `README.md` | This file |d


## Setup

1. **Clone the repo**
```bash
   git clone https://github.com/<your-username>/yt-rag-chatbot.git
   cd yt-rag-chatbot
```

2. **Create a virtual environment**
```bash
   python -m venv venv
   venv\Scripts\Activate.ps1      # Windows PowerShell
   # source venv/bin/activate     # Mac/Linux
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Add your Groq API key**

   Create a `.env` file in the project root:
