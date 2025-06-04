import pandas as pd

def evaluate(qrels_path, retrieved_ids, top_k=10):
    qrels = pd.read_csv(qrels_path, sep='\t', names=["query_id", "iter", "doc_id", "relevance"])
    relevant_docs = set(qrels[qrels["relevance"] > 0]["doc_id"])

    retrieved_at_k = retrieved_ids[:top_k]
    retrieved_relevant = [doc for doc in retrieved_at_k if doc in relevant_docs]

    precision = len(retrieved_relevant) / top_k
    recall = len(retrieved_relevant) / len(relevant_docs)
    map_score = sum([(i + 1) / (rank + 1) for i, rank in enumerate(
        [i for i, doc in enumerate(retrieved_at_k) if doc in relevant_docs])]) / len(relevant_docs)

    return precision, recall, map_score