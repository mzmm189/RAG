from importlib import import_module
import faiss

vector_module = import_module("04_vector_representation")
chunk_module = import_module("03_chunking")

chunk_embeddings = vector_module.chunk_embeddings
chunks_df = chunk_module.chunks_df


def create_faiss_index(embeddings, chunks_dataframe):
    """
    Build a FAISS IndexFlatIP (Inner Product) index for normalized embeddings.
    For normalized vectors, inner product equals cosine similarity.
    """
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    metadata_list = chunks_dataframe[
        ["doc_id", "chunk_id", "text", "specialty", "description"]
    ].to_dict("records")

    return index, metadata_list


faiss_index, metadata = create_faiss_index(chunk_embeddings, chunks_df)

if __name__ == "__main__":
    print(f"Number of vectors stored in FAISS index: {faiss_index.ntotal}")
    print(f"Number of metadata entries: {len(metadata)}")
