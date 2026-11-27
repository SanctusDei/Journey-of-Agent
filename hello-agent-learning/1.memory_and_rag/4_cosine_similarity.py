from math import sqrt

def cosine_similarity(a: list[float], b: list[float]) -> float:
    
    if len(a) != len(b):
        raise ValueError("vector dimensions must match")
    
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    print(cosine_similarity([1, 0], [1, 0]))
    print(cosine_similarity([1, 0], [0, 1]))
    print(cosine_similarity([1, 0], [-1, 0]))
