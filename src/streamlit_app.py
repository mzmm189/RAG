from importlib import import_module
import streamlit as st

rag = import_module("07_prompting")

st.set_page_config(
    page_title="Medical Transcriptions RAG Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Medical Transcriptions RAG Assistant")
st.markdown(
    "Ask clinical questions based on **Kaggle Medical Transcriptions (mtsamples.csv)** dataset using **FAISS** semantic search & **Flan-T5**."
)

with st.sidebar:
    st.header("Pipeline Configuration")
    top_k = st.slider(
        "Top-K Chunks to Retrieve", min_value=1, max_value=10, value=3
    )
    max_tokens = st.slider(
        "Max New Answer Tokens", min_value=20, max_value=250, value=100
    )

question = st.text_area(
    "Medical Question:",
    placeholder="e.g., What treatment was given for chest pain?",
    height=100,
)

if st.button("Answer Question", type="primary") and question.strip():
    with st.spinner("Searching medical database and generating response..."):
        result = rag.rag_answer(
            question, top_k=top_k, max_new_tokens=max_tokens
        )

    st.subheader("Generated Answer")
    st.info(result["answer"])

    with st.expander("🔍 View Retrieved Medical Context & Sources"):
        for i, source in enumerate(result["retrieved"], start=1):
            st.markdown(
                f"**Source {i}** | **Specialty:** `{source.get('specialty', 'N/A')}` | **Similarity Score:** `{source['similarity_score']:.3f}`"
            )
            if source.get("description"):
                st.caption(f"Summary: {source['description']}")
            st.text_area(
                f"Chunk {i} Content",
                value=source["text"],
                height=150,
                key=f"source_{i}",
            )
            st.divider()
