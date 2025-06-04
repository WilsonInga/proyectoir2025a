import ir_datasets
import pandas as pd

dataset = ir_datasets.load("beir/arguana")

docs_data = []
for i, doc in enumerate(dataset.docs_iter(), start=1):
    if doc.text and doc.text.strip():
        docs_data.append({'ID': i, 'Doc_ID': doc.doc_id, 'Texto': doc.text})

df = pd.DataFrame(docs_data)
df.to_csv("data/corpus_arguana_texto_plano.csv", index=False)