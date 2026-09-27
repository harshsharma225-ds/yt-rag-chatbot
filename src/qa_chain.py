import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

load_dotenv()


def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """Build a RetrievalQA chain from a vectorstore, using Groq's Llama 3.1."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


if __name__ == "__main__":
    from transcript_loader import get_transcript, extract_video_id
    from chunking import chunk_transcript
    from vectorstore import store_exists, load_vectorstore, build_vectorstore

    test_url = input("Enter a YouTube URL: ")
    video_id = extract_video_id(test_url)

    if store_exists(video_id):
        print("Loading existing vectorstore...")
        vs = load_vectorstore(video_id)
    else:
        print("Building new vectorstore...")
        text = get_transcript(test_url)
        docs = chunk_transcript(text)
        vs = build_vectorstore(video_id, docs)

    chain = build_qa_chain(vs)

    while True:
        query = input("\nAsk a question about the video (or 'exit'): ")
        if query.lower() == "exit":
            break
        result = chain.invoke({"query": query})
        print(f"\nAnswer: {result['result']}")