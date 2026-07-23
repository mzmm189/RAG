from importlib import import_module
import time
from transformers import pipeline

retrieval_module = import_module("06_retrieve_context")
vector_module = import_module("04_vector_representation")

semantic_search = retrieval_module.semantic_search
hybrid_search = retrieval_module.hybrid_search
chunk_tfidf_search = vector_module.chunk_tfidf_search
build_context = retrieval_module.build_context

MODEL_NAME = "google/flan-t5-small"

# Load instruction-tuned text generation pipeline
try:
    generator = pipeline("text2text-generation", model=MODEL_NAME)
except Exception:
    generator = pipeline("text-generation", model=MODEL_NAME)


def execute_retrieval(query, top_k=3, search_type="hybrid", alpha=0.6):
    """Dispatch retrieval to hybrid, semantic (FAISS), or chunk TF-IDF based on search_type."""
    search_type_clean = (search_type or "hybrid").lower()
    if search_type_clean == "hybrid":
        return hybrid_search(query, top_k=top_k, alpha=alpha)
    elif search_type_clean == "tfidf":
        return chunk_tfidf_search(query, top_k=top_k)
    else:
        return semantic_search(query, top_k=top_k)


def build_prompt(question, context):
    """Construct prompt constraining answer generation strictly to retrieved medical context."""
    return (
        "Answer the question using only the medical context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def rag_answer(question, top_k=3, max_new_tokens=100, search_type="hybrid", alpha=0.6):
    """Full RAG pipeline: Query -> Context Retrieval -> Hugging Face Flan-T5 -> Answer."""
    retrieved = execute_retrieval(
        question, top_k=top_k, search_type=search_type, alpha=alpha
    )
    context = build_context(retrieved)
    prompt = build_prompt(question, context)

    generated = generator(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
    return {
        "question": question,
        "search_type": search_type,
        "alpha": alpha if search_type == "hybrid" else None,
        "retrieved": retrieved,
        "context": context,
        "answer": generated,
    }


def timed_rag_answer(
    question, top_k=3, max_new_tokens=100, search_type="hybrid", alpha=0.6
):
    """Execute RAG pipeline while measuring retrieval speed vs generation latency."""
    t0 = time.time()
    retrieved = execute_retrieval(
        question, top_k=top_k, search_type=search_type, alpha=alpha
    )
    t1 = time.time()

    context = build_context(retrieved)
    prompt = build_prompt(question, context)
    generated = generator(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
    t2 = time.time()

    return {
        "question": question,
        "search_type": search_type,
        "top_k": top_k,
        "retrieved": retrieved,
        "answer": generated,
        "retrieval_time_sec": round(t1 - t0, 4),
        "generation_time_sec": round(t2 - t1, 4),
        "total_time_sec": round(t2 - t0, 4),
    }


if __name__ == "__main__":
    q = "What treatment was given for chest pain?"
    res_hybrid = rag_answer(q, top_k=3, search_type="hybrid", alpha=0.6)
    print("Q:", res_hybrid["question"])
    print("Strategy:", res_hybrid["search_type"])
    print("A:", res_hybrid["answer"])

