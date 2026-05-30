"""Standalone helper for Azure OpenAI chat completion calls.

Reads credentials from environment variables. See `.env.example`.
"""
import os

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

OPENAI_API_BASE = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
MODEL_NAME = os.environ.get("AZURE_OPENAI_LLM_MODEL", "gpt-4o")


def make_openai_client(endpoint, api_key, api_version):
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_openai(client, model, query, temperature=0):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
    )
    return completion.choices[0].message.content


def make_gpt_4():
    client = make_openai_client(
        endpoint=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )

    def query_gpt_4(query, temperature=0):
        return query_openai(client, MODEL_NAME, query, temperature)

    return query_gpt_4


def get_response(question, temperature=0):
    query_func = make_gpt_4()
    return query_func(question, temperature)


if __name__ == "__main__":
    print(get_response("Hello"))
