import os
from importlib import import_module
import streamlit as st

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

rag = import_module("07_prompting")

st.set_page_config(
    page_title="Medical Transcriptions RAG Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Medical Transcriptions RAG Assistant")
st.caption(
    "Clinical Q&A system built on Kaggle Medical Transcriptions using FAISS, TF-IDF Hybrid Search, OpenRouter LLM API & Local Flan-T5 Fallback."
)

st.divider()


# Quick sample questions
sample_questions = [
    "What treatment was given for chest pain?",
    "What surgical procedure was performed on the knee?",
    "What was the diagnosis for abdominal pain?",
    "What symptoms did the patient with a headache report?",
]

if "user_query" not in st.session_state:
    st.session_state["user_query"] = "What treatment was given for chest pain?"

st.subheader("📝 Clinical Query Input")

st.markdown("**Quick Examples:** Click any sample below to load into the question box:")
cols = st.columns(len(sample_questions))
for i, sample in enumerate(sample_questions):
    if cols[i].button(f"Sample {i+1}", key=f"sample_{i}"):
        st.session_state["user_query"] = sample

question = st.text_area(
    "Medical Question:",
    value=st.session_state["user_query"],
    placeholder="e.g., What treatment was given for chest pain?",
    height=120,
)

if st.button("🚀 Answer Question", type="primary") and question.strip():
    with st.spinner("Searching medical database & generating response..."):
        result = rag.rag_answer(question, top_k=5)

    st.subheader("💡 Generated Clinical Answer")
    st.caption(f"🤖 **Model Provider:** `{result.get('llm_provider', 'Unknown')}`")
    st.success(result["answer"])

    st.subheader("🔍 View Retrieved Medical Context & Sources")
    for i, source in enumerate(result["retrieved"], start=1):
        score_info = f"Similarity Score: `{source['similarity_score']:.3f}`"
        if "hybrid_score" in source:
            score_info = (
                f"Hybrid Score: `{source['hybrid_score']:.3f}` | "
                f"Semantic: `{source['semantic_score']:.3f}` | "
                f"TF-IDF: `{source['tfidf_score']:.3f}`"
            )

        with st.expander(
            f"Source {i} | Specialty: {source.get('specialty', 'N/A')} | {score_info}",
            expanded=(i == 1),
        ):
            if source.get("description"):
                st.markdown(f"**Summary:** *{source['description']}*")
            st.code(source["text"], language="text")



