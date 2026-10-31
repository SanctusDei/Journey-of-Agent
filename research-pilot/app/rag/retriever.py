from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def retrieve_papers(
    paper_details: dict[str, dict],
    query: str,
    top_k: int = 3,
) -> list[dict]:
    
    """从已缓存的论文摘要中检索相关内容"""

    documents = [
        (paper_id, detail.get("abstract", "").strip())
        for paper_id, detail in paper_details.items()
        if isinstance(detail.get("abstract"), str)
        and detail["abstract"].strip()
    ]

    if not documents or not query.strip() or top_k <= 0:
        return []

    ids, abstracts = zip(*documents)

    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 3)
    )

    vectors = vectorizer.fit_transform(abstracts)
    query_vector =  vectorizer.transform([query])

    scores =(vectors @ query_vector.T).toarray().ravel() 

    indices = scores.argsort()[::-1][:top_k]

    return [
        {
            "paper_id": ids[i],
            "abstract": abstracts[i],
            "score": float(scores[i]),
        }
        for i in indices
        if scores[i] > 0  
    ]



def retrieve_papers_semantic(
    paper_details: dict[str, dict],
    query: str,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[dict]:
    
    """从已缓存的论文摘要中检索相关内容"""

    documents = [
        (paper_id, detail.get("abstract", "").strip())
        for paper_id, detail in paper_details.items()
        if isinstance(detail.get("abstract"), str)
        and detail["abstract"].strip()
    ]

    if not documents or not query.strip() or top_k <= 0:
        return []

    ids, abstracts = zip(*documents)

    abstracts_embeddings = model.encode(abstracts, normalize_embeddings=True)
    query_embeddings = model.encode(query, normalize_embeddings=True)

    scores = abstracts_embeddings @ query_embeddings 

    indices = scores.argsort()[::-1][:top_k]

    return [
        {
            "paper_id": ids[i],
            "abstract": abstracts[i],
            "score": float(scores[i]),
        }
        for i in indices
    ]


if __name__ == "__main__":
     
    paper_details = {
        "W001": {
            "abstract": "毫米波雷达使用 Transformer 识别动态手势。"
        },
        "W002": {
            "abstract": "EEG 信号使用图神经网络识别情绪。"
        },
        "W003": {
            "abstract": "毫米波手势识别使用卷积神经网络。"
        },
    }

    results = retrieve_papers(
        paper_details,
        query="毫米波手势识别",
        top_k=2,
    )

    for item in results:
        print(item)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )
    paper_details = {
        "https://openalex.org/W1": {
            "abstract": "Fixing a broken automobile."
        },
        "https://openalex.org/W2": {
            "abstract": "Growing tomatoes in a garden."
        },
    }

    results = retrieve_papers_semantic(
        paper_details,
        query="car repair",
        model=model,
        top_k=2,
    )

    for item in results:
        print(item)
