from importlib import import_module
import streamlit as st

rag = import_module("07_prompting")

st.set_page_config(
    page_title="Medical Transcriptions RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🩺 Medical Transcriptions RAG Assistant")
st.caption(
    "Clinical Q&A system built on Kaggle Medical Transcriptions using FAISS, TF-IDF Hybrid Search, and Hugging Face Flan-T5."
)

st.divider()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")

    strategy_label = st.selectbox(
        "Retrieval Strategy",
        options=[
            "Hybrid (Lexical + Semantic)",
            "Semantic (FAISS Dense Search)",
            "TF-IDF (Lexical Search)",
        ],
        index=0,
        help="Select how medical documents are retrieved from the knowledge base.",
    )

    strategy_map = {
        "Hybrid (Lexical + Semantic)": "hybrid",
        "Semantic (FAISS Dense Search)": "semantic",
        "TF-IDF (Lexical Search)": "tfidf",
    }
    search_type = strategy_map[strategy_label]

    alpha = 0.6
    if search_type == "hybrid":
        st.markdown("**Hybrid Balance Weight (α)**")
        alpha = st.slider(
            "Semantic Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Alpha = 1.0 is pure Semantic search; Alpha = 0.0 is pure Lexical TF-IDF search.",
        )
        st.caption(
            f"🧬 Semantic Weight: `{alpha:.2f}` | 🔤 Lexical Weight: `{1-alpha:.2f}`"
        )
        st.divider()

    top_k = st.slider(
        "Top-K Chunks to Retrieve", min_value=1, max_value=10, value=3
    )
    max_tokens = st.slider(
        "Max New Answer Tokens", min_value=20, max_value=250, value=100
    )

    st.info(
        "💡 **Hybrid Search Tip:** Hybrid search combines exact keyword terms (e.g. drug names, dosages) with dense semantic embeddings."
    )

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
    with st.spinner("Searching medical database & generating response with Flan-T5..."):
        result = rag.rag_answer(
            question,
            top_k=top_k,
            max_new_tokens=max_tokens,
            search_type=search_type,
            alpha=alpha,
        )

    st.subheader("💡 Generated Clinical Answer")
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



