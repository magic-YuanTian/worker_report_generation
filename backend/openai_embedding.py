"""Standalone helper for Azure OpenAI embedding calls.

Reads credentials from environment variables. See `.env.example`.
"""
import os

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-15-preview")
MODEL_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")


def make_openai_client(endpoint, api_key, api_version):
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_openai(client, model, query):
    response = client.embeddings.create(input=query, model=model)
    return response.data[0].embedding


def make_embedder():
    client = make_openai_client(
        endpoint=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )

    def embed(query):
        return query_openai(client, MODEL_NAME, query)

    return embed


def get_embedding(question):
    return make_embedder()(question)


if __name__ == "__main__":
    import time

    start_time = time.time()
    res = get_embedding("which model are you?")
    print(f"Time used: {time.time() - start_time:.2f} seconds")
    print(res)
