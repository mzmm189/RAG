from importlib import import_module
import numpy as np

vector_module = import_module("04_vector_representation")
faiss_module = import_module("05_create_faiss_index")

embedding_model = vector_module.embedding_model
faiss_index = faiss_module.faiss_index
metadata = faiss_module.metadata


def semantic_search(query, top_k=3, index=None, meta=None, model=None):
    """Embed query, search FAISS index, and return top_k matching chunks with similarity scores."""
    if index is None:
        index = faiss_index
    if meta is None:
        meta = metadata
    if model is None:
        model = embedding_model

    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)
    scores, indices = scores[0], indices[0]

    results = []
    for score, idx in zip(scores, indices):
        item = meta[idx].copy()
        item["similarity_score"] = float(score)
        results.append(item)
    return results


def build_context(retrieved_chunks):
    """Join retrieved chunks into a context string labeled with source specialty/number."""
    parts = []
    for i, r in enumerate(retrieved_chunks, start=1):
        specialty = r.get("specialty") or "General"
        label = f"[Source {i} | specialty: {specialty}]"
        parts.append(f"{label}\n{r['text']}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    test_query = "chest pain and shortness of breath"
    retrieved = semantic_search(test_query, top_k=3)
    context_str = build_context(retrieved)
    print("Semantic Search Results:")
    for r in retrieved:
        print(f"  Score: {r['similarity_score']:.3f} | Specialty: {r['specialty']}")
    print("\nFormatted Context String:\n", context_str[:300], "...")
