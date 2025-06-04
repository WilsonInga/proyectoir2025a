from flask import Flask, request, render_template_string
from preprocessing import preprocess
from retrieval import tfidf_search
import pandas as pd

app = Flask(__name__)
df = pd.read_csv("data/corpus_arguana_texto_plano.csv")
docs = df["Texto"].tolist()

HTML = """
<form method="post">
  <input name="query" style="width: 300px;">
  <input type="submit">
</form>
{% for i, score, doc in results %}
  <p><b>{{ i }} - {{ score }}</b></p>
  <p>{{ doc }}</p>
  <hr>
{% endfor %}
"""

@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        query_pre = preprocess(query)
        ranked_indices, scores = tfidf_search(query_pre, docs)
        results = [(i, f"{scores[i]:.3f}", docs[i]) for i in ranked_indices[:10]]
    return render_template_string(HTML, results=results)

if __name__ == "__main__":
    app.run(debug=True)