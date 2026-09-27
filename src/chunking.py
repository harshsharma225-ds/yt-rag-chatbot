from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def chunk_transcript(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    """
    Split a transcript's full text into overlapping chunks, wrapped as
    LangChain Document objects (needed for embedding/vectorstore steps).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents


if __name__ == "__main__":
    from transcript_loader import get_transcript

    test_url = input("Enter a YouTube URL: ")
    text = get_transcript(test_url)
    docs = chunk_transcript(text)

    print(f"\nTotal chunks: {len(docs)}\n")
    print("First chunk preview:\n")
    print(docs[0].page_content)