"""Day 5: compare lexical matching with pretrained sentence embeddings."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()

documents = [
    "Fixing a broken automobile.",
    "Growing tomatoes in a garden.",
    "Training neural networks on images.",
]
# query = "car repair"
query = "automobile"

# Word-level TF-IDF makes the vocabulary mismatch easy to observe.
vectorizer = TfidfVectorizer()
document_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])
tfidf_scores = (document_vectors @ query_vector.T).toarray().ravel()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
document_embeddings = model.encode(documents, normalize_embeddings=True)
query_embedding = model.encode(query, normalize_embeddings=True)
embedding_scores = document_embeddings @ query_embedding

console.print("Embedding shape:", document_embeddings.shape)
for text, lexical, semantic in zip(documents, tfidf_scores, embedding_scores):
    console.print(f"TF-IDF={lexical:.3f} | Embedding={semantic:.3f} | {text}")

"""
TF-IDF -> 字面重合 -> 缺少字面重合时容易漏掉

Embedding -> 语义相关性 -> 有机会识别为相关

"""