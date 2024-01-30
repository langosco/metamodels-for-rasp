import jax
import jax.numpy as jnp
import numpy as np
from typing import Sequence, Iterator


def data_iterator(
        data,
        batchsize=1024, 
        skip_last=False,
        stacked_tree=False,
    ) -> Iterator:
    """Iterate over the data in batches. Single epoch."""
    n = tree_leaves_len(data) if stacked_tree else len(data)
    for i in range(0, n, batchsize):
        if i + batchsize > n:
            if skip_last:
                break
            else:
                batchsize = n - i

        if stacked_tree:
            yield jax.tree_map(lambda x: x[i:i + batchsize], data)
        else:
            yield data[i:i + batchsize]


def count_params(params: dict) -> int:
    """Count the number of parameters in a pytree of parameters."""
    return sum([x.size for x in jax.tree_util.tree_leaves(params)])


def tree_stack(trees):
    """Stacks a list of trees into a single tree with an extra dimension."""
    return jax.tree_map(lambda *x: jnp.stack(x), *trees)


def tree_leaves_len(tree):
    """Return the length of the first axis of the leaves of a pytree,
    assuming all leaves share the first axis."""
    axis_lengths = jax.tree_map(lambda x: x.shape[0], tree)
    axis_lengths = jax.tree_leaves(axis_lengths)
    assert all([x == axis_lengths[0] for x in axis_lengths]), (
        "All tree leaves must have the same length.")
    return axis_lengths[0]


def tree_unstack(tree):
    """Unstacks a tree with an extra dimension into a list of trees."""
    axlen = tree_leaves_len(tree)
    return [jax.tree_map(lambda x: x[i], tree) for i in range(axlen)]


def tree_to_numpy(tree):
    """Converts a tree of arrays to a tree of numpy arrays."""
    return jax.tree_map(lambda x: np.array(x), tree)


def flatten_dict(d: dict, parent_key: str = '', sep: str = '__') -> dict:
    """Maps a nested dict to a flat dict by concatenating the keys."""
    flat = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat[new_key] = v
    return flat


def get_activation_stats(x):
    return {"std": x.std(), "l1": jnp.abs(x).mean()}


def get_mean_and_std_of_tree(tree):
    """Get the mean and std of a pytree of arrays."""
    flat = jax.flatten_util.ravel_pytree(tree)[0]
    return flat.mean(), flat.std()


def split_data(data: Sequence, val_data_ratio: float = 0.1):
    split_index = int(len(data)*(1-val_data_ratio))
    return data[:split_index], data[split_index:]



color_codes = {
    "green": "\033[32m",
    "red": "\033[91m",
}

CLOSE_SEQ = "\033[0m"


def format_text(text: str, color=None, bold=False):
    if color is None:
        text = text
    else:
        text = f"{color_codes[color]}{text}"
    
    if bold:
        text = f"\033[1m{text}"
    
    return f"{text}{CLOSE_SEQ}"


def color_sequence(sequence: list[str], correct: list[bool]):
    """Color a sequence of strings according to whether they are correct."""
    assert len(sequence) == len(correct)
    return [
        format_text(s, color="green" if c else "red", bold=True) 
        for s, c in zip(sequence, correct)
    ]