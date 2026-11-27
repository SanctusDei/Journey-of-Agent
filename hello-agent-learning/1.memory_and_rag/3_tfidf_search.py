from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    
    documents = [
        "Python dependency management with uv",
        "Python unit testing with pytest",
        "Binary tree traversal algorithm" 
    ]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    query = vectorizer.transform(["Python pytest"])
    scores = cosine_similarity(query, vectors)[0]

    for doc, score in sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
    ):
        print(f"{score:.3f} {doc}")

