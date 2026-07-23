import os
import pandas as pd

DATA_CANDIDATES = [
    "data/mtsamples.csv",
]


def find_column(columns, keyword):
    """Return the first column whose lowercased name contains `keyword`, or None."""
    for c in columns:
        if keyword in c.lower():
            return c
    return None


def load_documents():
    """Load and perform initial structure cleaning on the medical transcriptions dataset."""
    csv_path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)

    if csv_path is None:
        try:
            os.makedirs("data", exist_ok=True)
            exit_code = os.system(
                "kaggle datasets download -d tboyle10/medicaltranscriptions -p data --unzip"
            )
            if exit_code == 0 and os.path.exists("data/mtsamples.csv"):
                csv_path = "data/mtsamples.csv"
        except Exception as e:
            print("Kaggle API automatic download attempt failed:", e)

    if csv_path is None:
        raise FileNotFoundError(
            "Could not find mtsamples.csv. Please place it in data/mtsamples.csv or root directory."
        )

    df_raw = pd.read_csv(csv_path)

    text_col = find_column(df_raw.columns, "transcription")
    specialty_col = find_column(df_raw.columns, "specialty")
    description_col = find_column(df_raw.columns, "description")

    if text_col is None:
        raise ValueError(
            f"No column containing 'transcription' found in {list(df_raw.columns)}"
        )

    df = df_raw.copy()

    # Drop missing, empty, or exact duplicate transcriptions
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].str.len() > 0]
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    return df, text_col, specialty_col, description_col


df, TEXT_COL, SPECIALTY_COL, DESCRIPTION_COL = load_documents()

if __name__ == "__main__":
    print(f"Loaded {len(df)} documents.")
    print(f"TEXT_COL: {TEXT_COL}")
    print(f"SPECIALTY_COL: {SPECIALTY_COL}")
    print(f"DESCRIPTION_COL: {DESCRIPTION_COL}")
