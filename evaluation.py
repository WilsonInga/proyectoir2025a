
import pandas as pd

def evaluate(qrels_path, retrieved_ids, top_k=10):
    qrels = pd.read_csv(qrels_path, sep='\t', names=["query_id", "iter", "doc_id", "relevance"])
    relevant_docs = set(qrels[qrels["relevance"] > 0]["doc_id"].astype(str))

    retrieved_at_k = [str(doc_id) for doc_id in retrieved_ids[:top_k]] # Ensure retrieved_ids are strings and consider top_k
    retrieved_relevant = [doc for doc in retrieved_at_k if doc in relevant_docs]

    precision = len(retrieved_relevant) / top_k if top_k > 0 else 0.0
    if len(relevant_docs) == 0:
        recall = 0.0
        map_score = 0.0
    else:
        recall = len(retrieved_relevant) / len(relevant_docs)
        # Calculate MAP only if there are relevant documents
        if retrieved_relevant: # Check if any relevant documents were retrieved
             relevant_ranks = [i + 1 for i, doc in enumerate(retrieved_at_k) if doc in relevant_docs]
             # Calculate precision at each relevant rank and average them
             average_precision = sum([(i + 1) / rank for i, rank in enumerate(relevant_ranks)]) / len(relevant_docs)

             map_score = sum([(i + 1) / (rank + 1) for i, rank in enumerate(
                 [i for i, doc in enumerate(retrieved_at_k) if doc in relevant_docs])]) / len(relevant_docs)
        else:
            map_score = 0.0


    return precision, recall, map_score
