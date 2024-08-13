import asyncio
from functools import cache

import litellm
import numpy as np
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential


TOKEN_LIMIT = 8000


@cache
def _get_encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


async def batch_embed(model: str, text: list[str], batch_size=128) -> np.ndarray:
    if len(text) > batch_size:
        return np.concatenate(
            await asyncio.gather(
                *[
                    batch_embed(model, text[i : i + batch_size], batch_size=batch_size)
                    for i in range(0, len(text), batch_size)
                ]
            )
        )

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, max=30),
    ):
        with attempt:
            encoder = _get_encoder()
            text_tokens = encoder.encode_batch(text)
            text_truncated = [
                text
                if len(tokens) < TOKEN_LIMIT
                else encoder.decode(tokens[: TOKEN_LIMIT // 2] + tokens[-TOKEN_LIMIT // 2 :])
                for text, tokens in zip(text, text_tokens)
            ]
            embeddings = await litellm.aembedding(model=model, input=text_truncated)
    assert embeddings.data and len(embeddings.data) == len(text), "Error getting embeddings"
    data = sorted(embeddings.data, key=lambda x: x["index"])
    return np.array([d["embedding"] for d in data])
