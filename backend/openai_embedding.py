"""Tiny standalone test: get one embedding from OpenAI.

Reads `OPENAI_API_KEY` from the environment automatically. Run with:
    python openai_embedding.py
"""
from openai import OpenAI


def get_embedding(text: str, model: str = "text-embedding-3-large") -> list[float]:
    client = OpenAI()
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding


if __name__ == "__main__":
    import time

    start = time.time()
    vec = get_embedding("which model are you?")
    print(f"dim={len(vec)}  time={time.time() - start:.2f}s")
    print(vec[:8], "...")
