from importlib import import_module
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

doc_module = import_module("01_documents")
prep_module = import_module("02_preprocessing")
chunk_module = import_module("03_chunking")

df = doc_module.df
TEXT_COL = doc_module.TEXT_COL
chunks_df = chunk_module.chunks_df

MODEL_NAME = "all-MiniLM-L6-v2"

# Document-level TF-IDF Pipeline
df_tfidf = df.copy()
df_tfidf["text_tfidf"] = df_tfidf[TEXT_COL].apply(prep_module.preprocess_for_tfidf)

tfidf_vectorizer = TfidfVectorizer(max_features=20000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_tfidf["text_tfidf"])


def tfidf_search(query, top_k=3):
    """Classic TF-IDF retrieval: preprocess query and rank documents by cosine similarity."""
    query_processed = prep_module.preprocess_for_tfidf(query)
    query_vec = tfidf_vectorizer.transform([query_processed])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]

    results = df_tfidf.iloc[top_idx].copy()
    results["tfidf_score"] = scores[top_idx]
    return results


# Chunk-level TF-IDF Pipeline (for Chunk Retrieval & Hybrid Search)
chunks_tfidf_series = chunks_df["text"].apply(prep_module.preprocess_for_tfidf)
chunk_tfidf_vectorizer = TfidfVectorizer(max_features=20000)
chunk_tfidf_matrix = chunk_tfidf_vectorizer.fit_transform(chunks_tfidf_series)


def chunk_tfidf_search(query, top_k=3):
    """Chunk-level TF-IDF search returning top_k chunk metadata entries with tfidf_score."""
    query_processed = prep_module.preprocess_for_tfidf(query)
    query_vec = chunk_tfidf_vectorizer.transform([query_processed])
    scores = cosine_similarity(query_vec, chunk_tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        item = chunks_df.iloc[idx].to_dict()
        item["similarity_score"] = float(scores[idx])
        item["tfidf_score"] = float(scores[idx])
        results.append(item)
    return results


# Dense Embedding Model
embedding_model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts, model=None):
    """Generate unit-normalized dense embeddings for a list of texts."""
    if model is None:
        model = embedding_model
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


chunk_embeddings = generate_embeddings(chunks_df["text"].tolist())

if __name__ == "__main__":
    print("Document TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Chunk TF-IDF matrix shape:", chunk_tfidf_matrix.shape)
    print("Dense embeddings shape:", chunk_embeddings.shape)
    print(f"Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")

