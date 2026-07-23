from importlib import import_module
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

vector_module = import_module("04_vector_representation")
faiss_module = import_module("05_create_faiss_index")
prep_module = import_module("02_preprocessing")

embedding_model = vector_module.embedding_model
chunk_embeddings = vector_module.chunk_embeddings
chunk_tfidf_vectorizer = vector_module.chunk_tfidf_vectorizer
chunk_tfidf_matrix = vector_module.chunk_tfidf_matrix
faiss_index = faiss_module.faiss_index
metadata = faiss_module.metadata


def min_max_normalize(scores):
    """Normalize array of float scores to [0, 1] range."""
    scores = np.asarray(scores, dtype=float)
    min_score, max_score = scores.min(), scores.max()
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


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


def hybrid_search(query, top_k=3, alpha=0.6, meta=None, model=None):
    """
    Hybrid Search combining Lexical (TF-IDF) and Semantic (Embeddings) search:
    hybrid_score = alpha * norm_semantic_score + (1 - alpha) * norm_lexical_score
    """
    if meta is None:
        meta = metadata
    if model is None:
        model = embedding_model

    # 1. Lexical search (TF-IDF cosine similarity across all chunks)
    query_processed = prep_module.preprocess_for_tfidf(query)
    lexical_query_vec = chunk_tfidf_vectorizer.transform([query_processed])
    lexical_scores = cosine_similarity(lexical_query_vec, chunk_tfidf_matrix).flatten()

    # 2. Semantic search (Embedding cosine similarity across all chunks)
    query_embedding = model.encode([query], normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_embedding, chunk_embeddings).flatten()

    # 3. Min-Max normalization & weighted score combination
    norm_lexical = min_max_normalize(lexical_scores)
    norm_semantic = min_max_normalize(semantic_scores)
    hybrid_scores = alpha * norm_semantic + (1 - alpha) * norm_lexical

    # Top-K ranking
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = meta[idx].copy()
        item["similarity_score"] = float(hybrid_scores[idx])
        item["hybrid_score"] = float(hybrid_scores[idx])
        item["semantic_score"] = float(semantic_scores[idx])
        item["tfidf_score"] = float(lexical_scores[idx])
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

    print("--- Semantic Search Results ---")
    sem_retrieved = semantic_search(test_query, top_k=3)
    for r in sem_retrieved:
        print(f"  Score: {r['similarity_score']:.3f} | Specialty: {r['specialty']}")

    print("\n--- Hybrid Search Results (alpha=0.6) ---")
    hyb_retrieved = hybrid_search(test_query, top_k=3, alpha=0.6)
    for r in hyb_retrieved:
        print(
            f"  Hybrid Score: {r['hybrid_score']:.3f} (Semantic: {r['semantic_score']:.3f}, Lexical: {r['tfidf_score']:.3f}) | Specialty: {r['specialty']}"
        )

    context_str = build_context(hyb_retrieved)
    print("\nFormatted Hybrid Context String:\n", context_str[:300], "...")

