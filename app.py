import sys
import os
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from transcript_loader import get_transcript, extract_video_id
from chunking import chunk_transcript
from vectorstore import store_exists, load_vectorstore, build_vectorstore
from qa_chain import build_qa_chain

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")
st.title("🎥 YouTube RAG Chatbot")
st.write("Paste a YouTube link, then ask questions about the video.")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

url = st.text_input("YouTube video URL")

if st.button("Load Video"):
    try:
        video_id = extract_video_id(url)

        with st.spinner("Processing video..."):
            if store_exists(video_id):
                vs = load_vectorstore(video_id)
            else:
                text = get_transcript(url)
                docs = chunk_transcript(text)
                vs = build_vectorstore(video_id, docs)

            st.session_state.chain = build_qa_chain(vs)
            st.session_state.messages = []

        st.success("Video loaded! Ask your questions below.")
    except ValueError as e:
        st.error(str(e))

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if st.session_state.chain:
    question = st.chat_input("Ask a question about the video")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.chain.invoke({"input": question})
                answer = result["answer"]
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})