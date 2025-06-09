import ir_datasets
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descarga recursos necesarios para tokenización y stopwords
nltk.download('punkt')
nltk.download('stopwords')

# Carga modelo de spaCy para lematización
nlp = spacy.load("en_core_web_sm")

# Inicializa lista de stopwords y el stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Función para preprocesar texto: minúsculas, limpieza, tokenización, stemming y lematización
def preprocess(text):
    text = text.lower()                                # Convertir a minúsculas
    text = re.sub(r'[^a-z\s]', '', text)               # Eliminar caracteres no alfabéticos
    tokens = nltk.word_tokenize(text)                  # Tokenizar
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]  # Stemming y eliminar stopwords
    doc = nlp(' '.join(tokens))                        # Procesar con spaCy para lematización
    lemmatized = [token.lemma_ for token in doc]       # Obtener lemas
    return ' '.join(lemmatized)                        # Devolver texto final preprocesado

# Cargar conjunto de datos ArgumAna desde BEIR
dataset = ir_datasets.load("beir/arguana")
docs_data = []

# Iterar sobre los documentos, preprocesar y almacenar
for i, doc in enumerate(dataset.docs_iter(), start=1):
    if doc.text and doc.text.strip():
        docs_data.append({
            'ID': i,
            'Doc_ID': doc.doc_id,
            'Texto_original': doc.text,
            'Texto_preprocesado': preprocess(doc.text)
        })

# Guardar resultados en un archivo CSV
df = pd.DataFrame(docs_data)
df.to_csv("data/corpus_arguana_preprocesado.csv", index=False)
