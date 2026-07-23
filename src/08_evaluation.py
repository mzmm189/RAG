from importlib import import_module
import pandas as pd

doc_module = import_module("01_documents")
vector_module = import_module("04_vector_representation")
retrieval_module = import_module("06_retrieve_context")
prompting_module = import_module("07_prompting")

df = doc_module.df
SPECIALTY_COL = doc_module.SPECIALTY_COL
tfidf_search = vector_module.tfidf_search
semantic_search = retrieval_module.semantic_search
rag_answer = prompting_module.rag_answer
timed_rag_answer = prompting_module.timed_rag_answer

# Benchmark evaluation dataset
EVAL_QUERIES = [
    {
        "question": "What treatment was given for chest pain?",
        "expected_specialty": "Cardiovascular / Pulmonary",
    },
    {
        "question": "What surgical procedure was performed on the knee?",
        "expected_specialty": "Orthopedic",
    },
    {
        "question": "What was the diagnosis for abdominal pain?",
        "expected_specialty": "Gastroenterology",
    },
    {
        "question": "What symptoms did the patient with a headache report?",
        "expected_specialty": "Neurology",
    },
    {
        "question": "What was found during the physical examination of the heart?",
        "expected_specialty": "Cardiovascular / Pulmonary",
    },
    {
        "question": "What follow-up care was recommended after a cesarean delivery?",
        "expected_specialty": "Obstetrics / Gynecology",
    },
    {
        "question": "What did the radiology report show for the chest?",
        "expected_specialty": "Radiology",
    },
    {
        "question": "What was noted during a general surgery consultation?",
        "expected_specialty": "Surgery",
    },
]

TEST_QUESTIONS = [
    "What treatment was given for chest pain?",
    "What are the symptoms of diabetes mentioned in these reports?",
    "How was the patient's blood pressure managed?",
    "What medication dosage was prescribed for infection?",
    "What surgical procedure was performed on the knee?",
    "What lab results indicated kidney problems?",
    "How was a patient with asthma treated?",
    "What was the diagnosis for abdominal pain?",
    "What follow-up care was recommended after surgery?",
    "What symptoms did the patient with a headache report?",
    "What was found during the physical examination of the heart?",
    "What allergy information was recorded for the patient?",
]


def clean_specialty(s):
    return (s or "").strip().lower()


def precision_at_k(relevant_flags, k):
    return sum(relevant_flags[:k]) / k


def recall_at_k(relevant_flags, k, total_relevant):
    if total_relevant == 0:
        return 0.0
    return sum(relevant_flags[:k]) / total_relevant


def hit_rate_at_k(relevant_flags, k):
    return 1 if sum(relevant_flags[:k]) > 0 else 0


def reciprocal_rank(relevant_flags):
    for rank, is_relevant in enumerate(relevant_flags, start=1):
        if is_relevant:
            return 1 / rank
    return 0.0


def evaluate_semantic_retriever(eval_queries=None, k=5):
    """Compute retrieval quality metrics across benchmark queries using specialty matching as proxy relevance."""
    if eval_queries is None:
        eval_queries = EVAL_QUERIES

    specialty_counts = df[SPECIALTY_COL].apply(clean_specialty).value_counts()

    rows = []
    for item in eval_queries:
        question = item["question"]
        expected = clean_specialty(item["expected_specialty"])
        total_relevant = int(specialty_counts.get(expected, 0))

        results = semantic_search(question, top_k=k)
        retrieved_doc_ids = [r["doc_id"] for r in results]
        similarity_scores = [round(r["similarity_score"], 3) for r in results]
        relevant_flags = [clean_specialty(r["specialty"]) == expected for r in results]

        rows.append(
            {
                "query": question,
                "expected_specialty": item["expected_specialty"],
                "retrieved_doc_ids": retrieved_doc_ids,
                "similarity_scores": similarity_scores,
                "avg_similarity": round(
                    sum(similarity_scores) / len(similarity_scores), 3
                ),
                f"precision@{k}": round(precision_at_k(relevant_flags, k), 3),
                f"recall@{k}": round(
                    recall_at_k(relevant_flags, k, total_relevant), 5
                ),
                f"hit_rate@{k}": hit_rate_at_k(relevant_flags, k),
                "reciprocal_rank": round(reciprocal_rank(relevant_flags), 3),
            }
        )
    return pd.DataFrame(rows)


def compare_tfidf_vs_semantic(
    query="patient with high blood pressure and diabetes", top_k=3
):
    """Compare exact keyword match (TF-IDF) vs dense semantic search on the same query."""
    print(f"\n=== Comparing TF-IDF vs Semantic Search for: '{query}' ===")
    print("\n--- Top TF-IDF Results ---")
    tfidf_res = tfidf_search(query, top_k=top_k)
    for _, r in tfidf_res.iterrows():
        desc = r[doc_module.DESCRIPTION_COL] if doc_module.DESCRIPTION_COL else ""
        print(f"Score: {r['tfidf_score']:.3f} | {desc}")

    print("\n--- Top Semantic Search Results ---")
    sem_res = semantic_search(query, top_k=top_k)
    for r in sem_res:
        print(f"Score: {r['similarity_score']:.3f} | {r['description']}")


def run_timing_evaluation(questions=None):
    """Compare performance and latency of Top-3 vs Top-5 retrieval."""
    if questions is None:
        questions = TEST_QUESTIONS[:5]

    timing_rows = []
    for q in questions:
        timing_rows.append(timed_rag_answer(q, top_k=3))
        timing_rows.append(timed_rag_answer(q, top_k=5))

    timing_df = pd.DataFrame(timing_rows)
    summary = (
        timing_df.groupby("top_k")[
            ["retrieval_time_sec", "generation_time_sec", "total_time_sec"]
        ]
        .mean()
        .round(4)
    )
    return timing_df, summary


if __name__ == "__main__":
    print("--- Retrieval Performance Metrics (K=5) ---")
    eval_df = evaluate_semantic_retriever(k=5)
    print(eval_df[["query", "precision@5", "hit_rate@5", "reciprocal_rank"]])

    compare_tfidf_vs_semantic()

    print("\n--- Timing Evaluation Summary ---")
    _, timing_summary = run_timing_evaluation()
    print(timing_summary)
