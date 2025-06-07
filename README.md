# Sistema de Recuperación de Información (TF-IDF y BM25) con Interfaz Web

Este proyecto implementa un sistema completo de recuperación de información utilizando los métodos **TF-IDF** y **BM25**. Se permite la consulta tanto desde la línea de comandos como desde una interfaz web desarrollada con Flask. Además, se integra un módulo de evaluación que calcula métricas como precisión, recall y MAP, comparando los resultados con los datos de relevancia reales (`qrels.tsv`).

---

## 📁 Estructura del Proyecto

```
proyecto_rdi/
│
├── data/
│   ├── corpus_arguana_preprocesado.csv   # Corpus preprocesado
│   ├── queries.tsv                        # Queries originales
│   ├── queries_preprocessed.tsv          # Queries preprocesadas
│   └── qrels.tsv                          # Juicios de relevancia (query_id, doc_id)
│
├── preprocessing.py                      # Funciones para preprocesamiento de texto
├── retrieval.py                          # Recuperación con TF-IDF y BM25
├── evaluation.py                         # Cálculo de precisión, recall y MAP
├── cli_app.py                            # Aplicación en consola
├── web_app_auto_queryid_preprocessed.py # Interfaz web automática sin selector de query_id
├── generar_qrels_y_queries.py            # Script para generar queries.tsv y qrels.tsv desde BEIR
└── requirements.txt                      # Librerías necesarias
```

---

## 🚀 ¿Qué hace este proyecto?

- **Preprocesa** corpus y queries (minúsculas, eliminación de signos, stopwords, stemming, lematización).
- Permite realizar consultas vía:
  - `cli_app.py`: desde la consola.
  - `web_app_auto_queryid_preprocessed.py`: desde la web, detectando automáticamente el `query_id` evaluado.
- **Calcula métricas** de evaluación:
  - *Precisión*
  - *Recall*
  - *MAP* (Mean Average Precision)
- Usa métodos clásicos de recuperación de información:
  - TF-IDF (con `TfidfVectorizer` de Scikit-learn)
  - BM25 (implementado desde scikit-learn con `TfidfVectorizer` modificado)

---

## 🌐 Interfaz Web (Flask)

Inicia la aplicación con:

```bash
python web_app_auto_queryid_preprocessed.py
```

- Ingresa una consulta en lenguaje natural.
- Selecciona el método: `TF-IDF` o `BM25`.
- El sistema:
  1. Preprocesa tu consulta.
  2. Recupera los documentos más relevantes.
  3. Imprime las métricas de evaluación en consola (si la query coincide con las preprocesadas).
  4. Muestra el contenido de los documentos relevantes en pantalla.

---

## 🧪 Evaluación Automática

La evaluación se realiza automáticamente si se encuentra un `query_id` cuyo texto preprocesado coincide con la consulta preprocesada ingresada.

La consola muestra algo como:

```
¡DEBUG ACTIVADO!
Relevantes para query_id test-environment-xxx: {...}
Top recuperados: [...]
Coincidencias relevantes encontradas: [...]
[Evaluación] Query ID: test-environment-xxx
Precisión: 0.200 | Recall: 0.400 | MAP: 0.237
```

---

## ⚙️ Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

---

## 📌 Consideraciones Técnicas

- El corpus y las queries provienen del dataset [`beir/arguana`](https://github.com/beir-cellar/beir).
- Se incluye un script para generar `queries.tsv` y `qrels.tsv` automáticamente con `ir_datasets`.
- El preprocesamiento se basa en NLTK + spaCy (`en_core_web_sm`).
- La evaluación compara los documentos recuperados con los esperados según los archivos `qrels.tsv`.

---

## 👨‍💻 Autor

Desarrollado por: Anthony Reinoso - Wilson Inga - Sergio Vite  


---

¡Esperamos que este sistema te sea útil para explorar y experimentar con técnicas de recuperación de información clásicas!
