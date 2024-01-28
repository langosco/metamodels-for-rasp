import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def pad_and_chunk(arr: ArrayLike, chunk_size: int) -> jax.Array:
    pad_size = -len(arr) % chunk_size
    padded = jnp.pad(arr, (0, pad_size))
    chunks = padded.reshape(-1, chunk_size)
    return chunks

