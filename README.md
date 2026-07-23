
## 📁 Repository Structure

- `01_documents.py`: Loads the Kaggle Medical Transcriptions dataset (`mtsamples.csv`), detects key columns (`transcription`, `specialty`, `description`), and performs missing value/duplicate filtering.
- `02_preprocessing.py`: Implements medical-safe text cleaning (preserving clinical numbers, dosages, and units) and NLTK TF-IDF preprocessing (stopword removal, tokenization, lemmatization).
- `03_chunking.py`: Chunks long medical reports using a sliding window algorithm (400 words, 50-word overlap) and attaches metadata (`doc_id`, `chunk_id`, `specialty`, `description`).
- `04_vector_representation.py`: Generates TF-IDF vector matrices and dense semantic embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`).
- `05_create_faiss_index.py`: Constructs an in-memory `faiss.IndexFlatIP` (Inner Product / Cosine Similarity) index and metadata lookup table.
- `06_retrieve_context.py`: Performs FAISS vector similarity search and builds formatted context strings for LLM prompting.
- `07_prompting.py`: Integrates Hugging Face's `google/flan-t5-small` model to generate grounded answers using retrieved context.
- `08_evaluation.py`: Evaluates retrieval quality (Precision@K, Recall@K, Hit Rate, MRR), timing benchmarks, and compares classic TF-IDF vs dense semantic retrieval.
- `streamlit_app.py`: Interactive Streamlit web application UI for visual medical Q&A.

## 🚀 Running the Project

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running Modules Individually
You can test any step of the pipeline by executing its module:
```bash
python 01_documents.py
python 03_chunking.py
python 06_retrieve_context.py
python 07_prompting.py
python 08_evaluation.py
```

### Launching the Web UI
To launch the interactive Streamlit assistant:
```bash
streamlit run streamlit_app.py
```
