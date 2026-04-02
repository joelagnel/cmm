"""Shared test fixtures — mock OpenAI embeddings for offline testing."""
import pytest
from unittest.mock import patch, MagicMock
import hashlib


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Generate deterministic fake embeddings from text content.

    Uses a bag-of-words approach so semantically similar texts
    (sharing many words) produce similar vectors.
    """
    # Build a shared vocabulary from all words seen
    DIM = 1536
    embeddings = []
    for text in texts:
        raw = [0.0] * DIM
        words = text.lower().split()
        for word in words:
            # Hash each word to a set of dimensions and increment
            h = hashlib.md5(word.encode()).digest()
            for j in range(4):  # activate 4 dims per word
                idx = (h[j * 2] * 256 + h[j * 2 + 1]) % DIM
                raw[idx] += 1.0
        # Normalize
        norm = sum(x * x for x in raw) ** 0.5
        if norm > 0:
            embeddings.append([x / norm for x in raw])
        else:
            embeddings.append([0.0] * DIM)
    return embeddings


@pytest.fixture(autouse=True)
def mock_openai_embeddings():
    """Auto-mock OpenAI embeddings for all tests so they run offline."""
    with patch("src.store.vector_store.OpenAI") as MockOpenAI:
        mock_client = MagicMock()

        def fake_create(model, input):
            embeddings = _fake_embed(input if isinstance(input, list) else [input])
            mock_response = MagicMock()
            mock_response.data = []
            for emb in embeddings:
                item = MagicMock()
                item.embedding = emb
                mock_response.data.append(item)
            return mock_response

        mock_client.embeddings.create = fake_create
        MockOpenAI.return_value = mock_client
        yield mock_client
