from importlib import import_module
import pandas as pd

doc_module = import_module("01_documents")
prep_module = import_module("02_preprocessing")

df = doc_module.df
TEXT_COL = doc_module.TEXT_COL
SPECIALTY_COL = doc_module.SPECIALTY_COL
DESCRIPTION_COL = doc_module.DESCRIPTION_COL


def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split `text` into overlapping chunks of `chunk_size` words, moving forward
    by (chunk_size - overlap) words each time. Short text (<= chunk_size) is returned
    as a single chunk unchanged.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def build_chunks(
    df_data=None, text_col=None, specialty_col=None, description_col=None
):
    """Build chunk dataframe with metadata from medical document reports."""
    if df_data is None:
        df_data = df
    if text_col is None:
        text_col = TEXT_COL
    if specialty_col is None:
        specialty_col = SPECIALTY_COL
    if description_col is None:
        description_col = DESCRIPTION_COL

    df_cleaned = df_data.copy()

    # Apply medical cleaning
    df_cleaned[text_col] = df_cleaned[text_col].apply(
        prep_module.clean_medical_text
    )

    if description_col and description_col in df_cleaned.columns:
        df_cleaned[description_col] = (
            df_cleaned[description_col]
            .astype(str)
            .apply(prep_module.normalize_whitespace)
        )

    chunk_records = []
    for row_id, row in df_cleaned.reset_index(drop=True).iterrows():
        chunks = chunk_text(row[text_col], chunk_size=400, overlap=50)
        for chunk_id, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "doc_id": row_id,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "specialty": (
                        row[specialty_col]
                        if (specialty_col and specialty_col in row)
                        else None
                    ),
                    "description": (
                        row[description_col]
                        if (description_col and description_col in row)
                        else None
                    ),
                }
            )

    return pd.DataFrame(chunk_records)


chunks_df = build_chunks()

if __name__ == "__main__":
    print(f"Generated {len(chunks_df)} chunks from {len(df)} documents.")
    print("Sample chunk:")
    print(chunks_df.head(1).to_dict("records")[0])
