
def chunk_text(text: str, size: int, overlap: int):
    if not 0 <= overlap < size:
        raise ValueError("Invalid chunk parameters")
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start: end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


if __name__ == "__main__":
    
    print(chunk_text("ABCDEFGHIJ", 4, 1))
