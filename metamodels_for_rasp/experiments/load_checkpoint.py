from time import time
import os
import pprint
import json
from collections import defaultdict
from dataclasses import asdict
import pickle
import matplotlib.pyplot as plt

import jax
import chex
import numpy as np
import argparse
import orbax.checkpoint
from jax.typing import ArrayLike

from metamodels_for_rasp import on_cluster
from metamodels_for_rasp.train import Logger
from metamodels_for_rasp.logger_config import setup_logger, setup_data_logger
from metamodels_for_rasp.model import Transformer, TransformerConfig
from metamodels_for_rasp.utils import count_params, data_iterator, color_sequence, create_loss_fn, compute_fracs_correct_by_program

from decompile_tracr.dataset import config as dataset_config
from decompile_tracr.dataset import data_utils
from decompile_tracr.tokenizing import vocab


logger = setup_logger(__name__)
data_logger = setup_data_logger(logfile="test.log")
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def get_model(checkpoint_path: str):
    params, config = checkpointer.restore(checkpoint_path)
    config = {k: v for k, v in config.items() if v is not None}
    config['decode'] = False  # no caching for now
#    config['weight_len'] = int(MAX_WEIGHTS_LENGTH / config['emb_dim'])
    config = TransformerConfig(**config)
    model = Transformer(config)
    return model, params


def log_stuff(test_datasets, params, model):
    for test_data in test_datasets.values():
        logger.info("Number of test examples: "
                    f"{len(test_data['rasp_tok'])}.")
        logger.info(f"Test data (weights) mean: {test_data['weights'].mean()}, "
                    f"std: {test_data['weights'].std()}")
    logger.info(f"Number of parameters in meta-model: "
                f"{count_params(params) / 1e6} Million")
    logger.info(f"Percent of weights zero: "
                f"{round(100 * (test_data['weights'] == 0).sum() / test_data['weights'].size, 2)}%")
    logger.info(f"Expected sequence length: {model.config.max_len}.")


def log_rasp_snippet(
        batch: ArrayLike, 
        correct_preds: ArrayLike, 
        name: str,
        snip_at: int = 10,
        batch_element: int = 0,
    ):
    rasp_snippet = tokenizer.decode(batch['rasp_tok'][batch_element, :snip_at])
    rasp_snippet = color_sequence(rasp_snippet, correct_preds[:snip_at])
    data_logger.info(f"{name}: {rasp_snippet}")
    return None


def pass_over_test_set(data, loss_fn, params, bs=256, decode=False):
    """One forward pass over a test set."""
    out = dict(mask=[], correct_preds=[])
    dummy_rng = jax.random.PRNGKey(0)  # dropout is off
    for batch in data_iterator(data, bs, stacked_tree=True):
        if decode is None:
            loss, aux = loss_fn(params, dummy_rng, batch, is_training=False)
        else:
            aux = decode(params, batch)

        for k in out.keys():
            out[k].append(aux[k])

    out = {k: np.concatenate(v) for k, v in out.items()}
    chex.assert_equal_shape_prefix(list(out.values()) + [data['rasp_tok']], prefix_len=2)
    return out


categories = {
    "encodings": tokenizer.encode(vocab.encodings),
    "ops": tokenizer.encode(vocab.ops),
    "maps": tokenizer.encode(vocab.maps),
    "variables": tokenizer.encode(vocab.sop_variables + vocab.inputs),
    "comparisons": tokenizer.encode(vocab.comparisons),
    "Map": vocab.vocab.index("Map"),
    "SequenceMap (both)": tokenizer.encode(["SequenceMap", "LinearSequenceMap"]),
    "LinearSequenceMap": vocab.vocab.index("LinearSequenceMap"),
    "SequenceMap": vocab.vocab.index("SequenceMap"),
    "SelectAggregate": vocab.vocab.index("SelectAggregate"),
    "SelectorWidth": vocab.vocab.index("SelectorWidth"),
}


def compute_token_acc_for_category(
        token_category: chex.Array,
        correct_preds: chex.Array,
        true_tokens: chex.Array,
    ) -> float:
    """
    Args:
        token_category: 0-d array, 1-d array, or list of tokens
        correct_preds: (n, seq_len) array of correct predictions
        true_tokens: (n, seq_len) array of true tokens
    
    Returns:
        float: mean accuracy for the given token category
    """
    where_tokens = np.where(np.isin(true_tokens, token_category))
    relevant_preds = correct_preds[where_tokens]
    return np.mean(relevant_preds)


def compute_token_acc_for_all_categories(
        categories: dict,
        correct_preds: chex.Array,
        true_tokens: chex.Array,
    ) -> dict:
    """
    Args:
        categories: dict of token categories
        correct_preds: (n, seq_len) array of correct predictions
        true_tokens: (n, seq_len) array of true tokens
    
    Returns:
        dict: mean accuracy for each token category
    """
    return {
        k: compute_token_acc_for_category(v, correct_preds, true_tokens)
        for k, v in categories.items()
    }


def save_to_json(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Test a trained model.')

    # wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--autoregressive', action='store_true', 
                        help="Use autoregressive decoding.")

    args = parser.parse_args()

    if args.checkpoint_path is None:
        raise ValueError("Must specify checkpoint path.")

    args.tags.append("HPC" if on_cluster else "local")
    if args.wandb_run_name is not None:
        args.wandb_run_name += str(int(time()))
    logger.info("Args:\n%s", pprint.pformat(vars(args)))

    model, params = get_model(args.checkpoint_path)
    test_datasets = load_data(model.config.emb_dim)
    log_stuff(test_datasets, params, model)
    loss_fn = create_loss_fn(model.apply)

    
    print()
    for x in test_datasets.values():
        print("dataset keys:", x.keys())
    print()


    weight_len = model.config.weight_len
    rasp_tok_len = model.config.rasp_tok_len


    def forward(params: dict, batch: dict):
        dummy_rng = jax.random.PRNGKey(0)  # dropout is off
        out = model.apply(
            {'params': params},
            {k: v for k, v in batch.items()},
            is_training=False,
            rngs={'dropout': dummy_rng},
        )

        logits = out[:, weight_len-1:-1, :]
        return logits


    def get_next_pred(params: dict, batch: dict, i: int):
        """Get temperature=0 prediction for next token at
        position i in the sequence (not counting weights)."""
        logits = forward(params, batch)
        next_logits = logits[:, i, :]
        preds = next_logits.argmax(axis=-1).astype(np.int32)
        return preds


    def get_mask(tokens):
        """Get mask for padding tokens."""
        return np.where(tokens == vocab.pad_id, 0, 1)


    def decode(params, batch: dict):
        """Autoregressively decode the batch."""
        output = {
            "weights": batch["weights"],
            "rasp_tok": np.zeros(batch['rasp_tok'].shape).astype(int),
        }
        correct_preds = np.zeros(batch['rasp_tok'].shape).astype(int)
        for i in range(0, rasp_tok_len):
            if i == 0:
                output['rasp_tok'][:, i] = vocab.bos_id
            output['rasp_tok'][:, i] = get_next_pred(params, output, i)
            correct_preds[:, i] = output['rasp_tok'][:, i] == batch['rasp_tok'][:, i]
        
        #output = output['rasp_tok']
        output = {
            "correct_preds": correct_preds,
            "mask": get_mask(batch['rasp_tok']),
        }
        return output


    for name, data in test_datasets.items():
        logger.info(f"Evaluating dataset {name}...")
        if args.autoregressive:
            decode = decode
        else:
            decode = None
        m = pass_over_test_set(data, loss_fn=loss_fn, params=params, bs=256,
                               decode=decode)
        fracs_corr = compute_fracs_correct_by_program(
            program_ids=data['program_id'],
            correct_preds=m['correct_preds'],
            mask=m['mask'],
        )

        if name == 'lib':
            savepath = dataset_config.data_dir / "metrics" / "metrics-lib.pkl"
            os.makedirs(savepath.parent, exist_ok=True)
            logger.info(f"Saving metrics for lib dataset to {savepath}.")
            with open(savepath, 'wb') as f:
                pickle.dump(m, f)

        logger.info(f"Overall accuracy: {np.mean(fracs_corr)}")

        accs_by_category = compute_token_acc_for_all_categories(
            categories=categories,
            correct_preds=m['correct_preds'],
            true_tokens=data['rasp_tok'],
        )

        logger.info("Token accuracy by category:")
        for k, v in accs_by_category.items():
            logger.info(f"{k}: {v}")

        if not args.autoregressive:
            savepath = dataset_config.data_dir / 'metrics' / f"{name}-metrics-by-category.json"
        else:
            savepath = dataset_config.data_dir / 'metrics' / f"{name}-metrics-by-category-autoregressive.json"
        os.makedirs(savepath.parent, exist_ok=True)
        logger.info(f"Saving metrics to {savepath}.")
        save_to_json(accs_by_category, savepath)



if __name__ == "__main__":
    main()