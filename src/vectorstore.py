import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

VECTORSTORE_DIR = "../vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def video_store_path(video_id: str) -> str:
    return os.path.join(VECTORSTORE_DIR, video_id)


def store_exists(video_id: str) -> bool:
    return os.path.isdir(video_store_path(video_id))


def build_vectorstore(video_id: str, documents: list[Document]) -> Chroma:
    """Embed and persist documents for a given video ID."""
    embeddings = get_embeddings()
    persist_dir = video_store_path(video_id)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(video_id: str) -> Chroma:
    """Load an already-persisted vectorstore for a given video ID."""
    embeddings = get_embeddings()
    persist_dir = video_store_path(video_id)

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    from transcript_loader import get_transcript, extract_video_id
    from chunking import chunk_transcript

    test_url = input("Enter a YouTube URL: ")
    video_id = extract_video_id(test_url)

    if store_exists(video_id):
        print("Vectorstore already exists for this video. Loading it...")
        vs = load_vectorstore(video_id)
    else:
        print("Building new vectorstore...")
        text = get_transcript(test_url)
        docs = chunk_transcript(text)
        vs = build_vectorstore(video_id, docs)
        print(f"Stored {len(docs)} chunks.")

    query = input("\nTest query: ")
    results = vs.similarity_search(query, k=3)
    print("\nTop matches:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.page_content[:200]}...\n")