from importlib import import_module
import time
from transformers import pipeline

retrieval_module = import_module("06_retrieve_context")
semantic_search = retrieval_module.semantic_search
build_context = retrieval_module.build_context

MODEL_NAME = "google/flan-t5-small"

# Load instruction-tuned text generation pipeline
try:
    generator = pipeline("text2text-generation", model=MODEL_NAME)
except Exception:
    generator = pipeline("text-generation", model=MODEL_NAME)


def build_prompt(question, context):
    """Construct prompt constraining answer generation strictly to retrieved medical context."""
    return (
        "Answer the question using only the medical context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def rag_answer(question, top_k=3, max_new_tokens=100):
    """Full RAG pipeline: Query -> Semantic Retrieval -> Context -> Hugging Face Flan-T5 -> Answer."""
    retrieved = semantic_search(question, top_k=top_k)
    context = build_context(retrieved)
    prompt = build_prompt(question, context)

    generated = generator(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
    return {
        "question": question,
        "retrieved": retrieved,
        "context": context,
        "answer": generated,
    }


def timed_rag_answer(question, top_k=3, max_new_tokens=100):
    """Execute RAG pipeline while measuring retrieval speed vs generation latency."""
    t0 = time.time()
    retrieved = semantic_search(question, top_k=top_k)
    t1 = time.time()

    context = build_context(retrieved)
    prompt = build_prompt(question, context)
    generated = generator(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
    t2 = time.time()

    return {
        "question": question,
        "top_k": top_k,
        "retrieved": retrieved,
        "answer": generated,
        "retrieval_time_sec": round(t1 - t0, 4),
        "generation_time_sec": round(t2 - t1, 4),
        "total_time_sec": round(t2 - t0, 4),
    }


if __name__ == "__main__":
    q = "What treatment was given for chest pain?"
    res = rag_answer(q, top_k=3)
    print("Q:", res["question"])
    print("A:", res["answer"])
