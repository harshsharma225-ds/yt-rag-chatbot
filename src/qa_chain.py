import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

load_dotenv()

SYSTEM_PROMPT = (
    "You are an assistant answering questions about a YouTube video, "
    "using only the transcript context provided below. "
    "If the answer isn't in the context, say you don't know. "
    "Keep answers concise.\n\n"
    "Context:\n{context}"
)


def build_qa_chain(vectorstore: Chroma):
    """Build a retrieval + answer chain from a vectorstore, using Groq's Llama 3.1."""
    llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


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
        result = chain.invoke({"input": query})
        print(f"\nAnswer: {result['answer']}")