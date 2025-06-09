from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Función para buscar documentos usando el algoritmo TF-IDF
def tfidf_search(query, docs):
   # Inicializa un objeto TfidfVectorizer
    vect = TfidfVectorizer()
    # Aprende el vocabulario y el idf, y transforma los documentos en una matriz TF-IDF
    tfidf_matrix = vect.fit_transform(docs)
    # Transforma la consulta en un vector TF-IDF utilizando el vocabulario aprendido
    query_vec = vect.transform([query])
    # Calcula la similitud del coseno entre el vector de consulta y cada vector de documento
    # .flatten() convierte la matriz de similitud resultante en un array 1D
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Devuelve los índices de los documentos ordenados por puntuación en orden descendente
    # y las puntuaciones originales
    return np.argsort(scores)[::-1], scores

# Función para buscar documentos usando el algoritmo BM25
# k1 y b son parámetros de ajuste del algoritmo BM25
def bm25_search(query, docs, k1=1.5, b=0.75):
    # Importación de bibliotecas necesarias dentro de la función (aunque ya están importadas arriba)
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    # Inicializa un objeto CountVectorizer para contar las ocurrencias de términos
    vectorizer = CountVectorizer()
    # Cuenta la frecuencia de los términos en cada documento y crea una matriz término-documento
    X = vectorizer.fit_transform(docs)
    # Obtiene la lista de términos (vocabulario) aprendidos por el vectorizador
    terms = vectorizer.get_feature_names_out()

    # Preprocesamiento simple de la consulta para BM25 (convertir a minúsculas y dividir)
    query_terms = query.lower().split()
    # Calcula la longitud (número de términos) de cada documento
    doc_len = X.sum(axis=1)
    # Calcula la longitud promedio de los documentos
    avg_len = doc_len.mean()
    # Inicializa una lista para almacenar las puntuaciones BM25 de cada documento
    scores = []

    # Itera a través de cada documento en el corpus
    for i in range(X.shape[0]):
        score = 0 # Inicializa la puntuación BM25 para el documento actual
        # Itera a través de cada término en la consulta
        for term in query_terms:
            # Comprueba si el término de la consulta está en el vocabulario de los documentos
            if term in terms:
                # df: Document Frequency (número de documentos que contienen el término)
                df = np.count_nonzero(X[:, vectorizer.vocabulary_[term]].toarray())
                # idf: Inverse Document Frequency (una medida de cuán raro es el término)
                idf = np.log((X.shape[0] - df + 0.5) / (df + 0.5) + 1)
                # tf: Term Frequency (número de veces que el término aparece en el documento actual)
                tf = X[i, vectorizer.vocabulary_[term]]
                # denom: Denominador en la fórmula BM25 que normaliza la TF
                denom = tf + k1 * (1 - b + b * doc_len[i, 0] / avg_len)
                # Calcula la parte de la puntuación BM25 para el término actual y la agrega a la puntuación total del documento
                score += idf * (tf * (k1 + 1)) / denom
        # Agrega la puntuación total del documento a la lista de puntuaciones
        scores.append(score)
    # Devuelve los índices de los documentos ordenados por puntuación en orden descendente
    # y las puntuaciones originales
    return np.argsort(scores)[::-1], scores
