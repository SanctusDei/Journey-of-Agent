from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    from rich.console import Console
    
    console = Console()
    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2", device="cpu"
    )

    texts = [
        "使用 uv 管理 Python 依赖",
        "如何安装 Python 软件包",
        "二叉树的中序遍历"
    ]

    embeddings = model.encode(texts)
    console.print(embeddings.shape)
    console.print(model.similarity(embeddings, embeddings))
