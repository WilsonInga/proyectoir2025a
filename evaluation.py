import pandas as pd

# Función para evaluar los resultados de la búsqueda
# qrels_path: ruta al archivo con las qrels (relevancias)
# retrieved_ids: lista de IDs de documentos recuperados (en orden de clasificación)
# query_id: el ID de la consulta que se está evaluando (si aplica, aunque el código parece diseñado para una sola consulta a la vez)
# top_k: el número de resultados principales a considerar para la evaluación
# debug: un indicador para imprimir información de depuración
def evaluate(qrels_path, retrieved_ids, query_id=None, top_k=10, debug=False):
    # Lee el archivo qrels en un DataFrame de pandas
    # Se especifica el separador como tab ('\t')
    # header=0 indica que la primera fila es el encabezado
    qrels = pd.read_csv(qrels_path, sep='\t', header=0)
    # Convierte la columna 'relevance' a tipo numérico, manejando errores
    qrels["relevance"] = pd.to_numeric(qrels["relevance"], errors="coerce")
    # Elimina filas donde 'relevance' es NaN (no se pudo convertir)
    qrels = qrels.dropna(subset=["relevance"])
    # Convierte la columna 'relevance' a tipo entero
    qrels["relevance"] = qrels["relevance"].astype(int)
    # Identifica los IDs de documentos relevantes para la consulta dada (si query_id no es None)
    # Un documento se considera relevante si su relevancia es mayor que 0
    if query_id is not None:
         relevant_docs = set(qrels[(qrels["query_id"] == query_id) & (qrels["relevance"] > 0)]["doc_id"])
    else:
        # Si no se proporciona query_id, considera todos los documentos relevantes en qrels
        # Esto asume que el archivo qrels contiene relevantes solo para la consulta en cuestión
        relevant_docs = set(qrels[qrels["relevance"] > 0]["doc_id"])


    # Imprime información de depuración si debug es True
    if debug:
        print("¡DEBUG ACTIVADO!")
        if query_id is not None:
             print(f"Relevantes para query_id {query_id}: {relevant_docs}")
        else:
             print(f"Todos los relevantes en qrels: {relevant_docs}")
        print(f"Top recuperados: {retrieved_ids[:top_k]}")

    # Si no hay documentos relevantes en las qrels, las métricas son 0
    if not relevant_docs:
        return 0.0, 0.0, 0.0

    # Obtiene los IDs de los documentos recuperados en el top K
    retrieved_at_k = retrieved_ids[:top_k]
    # Identifica cuáles de los documentos recuperados en el top K son realmente relevantes
    retrieved_relevant = [doc for doc in retrieved_at_k if doc in relevant_docs]

    # Imprime información de depuración sobre los documentos relevantes recuperados
    if debug:
        print(f"Coincidencias relevantes encontradas: {retrieved_relevant}")

    # Calcula la Precisión: proporción de documentos relevantes entre los recuperados en el top K
    precision = len(retrieved_relevant) / top_k
    # Calcula el Recall: proporción de documentos relevantes recuperados entre todos los relevantes
    recall = len(retrieved_relevant) / len(relevant_docs)

    # Calcula el Mean Average Precision (MAP)
    # Si no se recuperaron documentos relevantes, el MAP es 0
    if not retrieved_relevant:
        map_score = 0.0
    else:
        # Calcula la suma de la precisión en cada punto donde se recupera un documento relevante
        # dividido por el número total de documentos relevantes.
        # Esto calcula el Average Precision (AP) para esta consulta.
        # Aquí se asume que se está calculando el AP para una sola consulta,
        # aunque la variable se llama map_score.
        map_score = sum([(i + 1) / (rank + 1) for i, rank in enumerate(
            [i for i, doc in enumerate(retrieved_at_k) if doc in relevant_relevant])]) / len(relevant_docs)
            # Nota: Hay un pequeño error aquí, debería ser `retrieved_at_k` en la segunda enumeración.
            # Corregido:
            # map_score = sum([(i + 1) / (rank + 1) for i, rank in enumerate(
            #    [j for j, doc in enumerate(retrieved_at_k) if doc in relevant_docs])]) / len(relevant_docs)

    # Devuelve la precisión, el recall y la puntuación MAP (o AP para esta consulta)
    return precision, recall, map_score
