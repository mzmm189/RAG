from importlib import import_module
import json
import os
import time
import urllib.error
import urllib.request
from transformers import pipeline

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

retrieval_module = import_module("06_retrieve_context")
vector_module = import_module("04_vector_representation")

semantic_search = retrieval_module.semantic_search
hybrid_search = retrieval_module.hybrid_search
chunk_tfidf_search = vector_module.chunk_tfidf_search
build_context = retrieval_module.build_context

MODEL_NAME = "google/flan-t5-small"


def get_openrouter_api_key():
    """Retrieve OpenRouter API key from environment variables or .env file."""
    return (
        os.getenv("OPEN_ROUTER_KEY", "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
    )


OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Load instruction-tuned text generation pipeline as local fallback
try:
    generator = pipeline("text2text-generation", model=MODEL_NAME)
except Exception:
    generator = pipeline("text-generation", model=MODEL_NAME)


def generate_with_openrouter(prompt, max_tokens=100, model=None, api_key=None):
    """Call OpenRouter API to generate answer using standard library urllib."""
    key = api_key or get_openrouter_api_key()
    if not key:
        raise ValueError("OPENROUTER_API_KEY (or OPEN_ROUTER_KEY) is not set.")

    target_model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "Health Doc RAG Assistant",
        "User-Agent": "Python-Urllib/3.0",
    }
    payload = {
        "model": target_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful clinical assistant. Answer questions strictly using the provided medical context.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            return res_json["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8") if e.fp else str(e)
        raise RuntimeError(f"OpenRouter API returned HTTP {e.code}: {err_body}")
    except Exception as e:
        raise RuntimeError(f"OpenRouter connection error: {e}")


def generate_text_with_fallback(prompt, max_new_tokens=100, api_key=None, model=None):
    """Generate text using OpenRouter API, falling back to local Flan-T5 if unavailable."""
    try:
        ans = generate_with_openrouter(
            prompt, max_tokens=max_new_tokens, model=model, api_key=api_key
        )
        used_model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        return ans, f"OpenRouter API ({used_model})"
    except Exception as err:
        # Fallback to local Hugging Face model
        print(f"[Fallback Notice] OpenRouter unavailable ({err}). Using local model ({MODEL_NAME})...")
        generated = generator(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
        return generated, f"Local HF ({MODEL_NAME})"


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


def rag_answer(
    question,
    top_k=3,
    max_new_tokens=100,
    search_type="hybrid",
    alpha=0.6,
    api_key=None,
    model=None,
):
    """Full RAG pipeline: Query -> Context Retrieval -> OpenRouter (or Local Flan-T5 fallback) -> Answer."""
    retrieved = execute_retrieval(
        question, top_k=top_k, search_type=search_type, alpha=alpha
    )
    context = build_context(retrieved)
    prompt = build_prompt(question, context)

    generated, provider = generate_text_with_fallback(
        prompt, max_new_tokens=max_new_tokens, api_key=api_key, model=model
    )
    return {
        "question": question,
        "search_type": search_type,
        "alpha": alpha if search_type == "hybrid" else None,
        "retrieved": retrieved,
        "context": context,
        "answer": generated,
        "llm_provider": provider,
    }


def timed_rag_answer(
    question,
    top_k=3,
    max_new_tokens=100,
    search_type="hybrid",
    alpha=0.6,
    api_key=None,
    model=None,
):
    """Execute RAG pipeline while measuring retrieval speed vs generation latency."""
    t0 = time.time()
    retrieved = execute_retrieval(
        question, top_k=top_k, search_type=search_type, alpha=alpha
    )
    t1 = time.time()

    context = build_context(retrieved)
    prompt = build_prompt(question, context)
    generated, provider = generate_text_with_fallback(
        prompt, max_new_tokens=max_new_tokens, api_key=api_key, model=model
    )
    t2 = time.time()

    return {
        "question": question,
        "search_type": search_type,
        "top_k": top_k,
        "retrieved": retrieved,
        "answer": generated,
        "llm_provider": provider,
        "retrieval_time_sec": round(t1 - t0, 4),
        "generation_time_sec": round(t2 - t1, 4),
        "total_time_sec": round(t2 - t0, 4),
    }


if __name__ == "__main__":
    q = "What treatment was given for chest pain?"
    res_hybrid = rag_answer(q, top_k=3, search_type="hybrid", alpha=0.6)
    print("Q:", res_hybrid["question"])
    print("Strategy:", res_hybrid["search_type"])
    print("Provider:", res_hybrid["llm_provider"])
    print("A:", res_hybrid["answer"])

