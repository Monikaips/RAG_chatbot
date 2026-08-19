import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from rag.openai_usage import record_from_openai_response


load_dotenv()


class TrackedOpenAIEmbeddings(Embeddings):
    """
    OpenAI embeddings that record actual usage from each API response.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from .env")

        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.chunk_size = 100

    def embed_documents(self, texts):
        embeddings = []

        for index in range(0, len(texts), self.chunk_size):
            batch = texts[index:index + self.chunk_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            record_from_openai_response(response, model=self.model)
            embeddings.extend(item.embedding for item in response.data)

        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def get_embeddings():
    return TrackedOpenAIEmbeddings(
        model="text-embedding-3-small"
    )
