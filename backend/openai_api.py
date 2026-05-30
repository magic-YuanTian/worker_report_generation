"""Tiny standalone test: send one chat completion to OpenAI.

Reads `OPENAI_API_KEY` from the environment automatically. Run with:
    python openai_api.py
"""
from openai import OpenAI


def get_response(question: str, model: str = "gpt-4o", temperature: float = 0) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    print(get_response("Hello"))
