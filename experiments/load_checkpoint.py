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
from metamodels_for_rasp.utils import count_params, data_iterator, color_sequence, create_loss_fn, get_fracs_correct_by_program

from rasp_tokenizer import paths
from rasp_tokenizer.data_utils import load_and_process_data
from rasp_tokenizer import MAX_RASP_LENGTH, MAX_WEIGHTS_LENGTH
from rasp_tokenizer import tokenizer


logger = setup_logger(__name__)
data_logger = setup_data_logger(logfile="test.log")
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


TEST_DATA_NAMES = [
#    "train",
    "test_5",
    "test_6",
    "test_10",
    "test",
    "lib",
]


def get_data(d_model: int):
    TRAIN_DATA_STD = 2.656

    test_datasets = {
        name: load_and_process_data(
            rng=None,
            ndata=None,
            shuffle=False,
            name=name,
            d_model=d_model,
            max_rasp_len=MAX_RASP_LENGTH,
            max_weights_len=MAX_WEIGHTS_LENGTH,
        ) for name in TEST_DATA_NAMES
    }

    for v in test_datasets.values():
        v['weights'] = v['weights'] / TRAIN_DATA_STD

    seq_len=MAX_RASP_LENGTH + MAX_WEIGHTS_LENGTH/d_model
    assert seq_len.is_integer()
    seq_len = int(seq_len)

    return test_datasets, seq_len


def get_model(checkpoint_path: str):
    params, config = checkpointer.restore(checkpoint_path)
    config = {k: v for k, v in config.items() if v is not None}
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


def get_program_list(metrics: dict[str, ArrayLike]) -> list[list[dict]]:
    """Convert dict of metrics into list of lists of dicts, one list 
    per program. That is, 
    - the output is a list of programs, where 
    - each program is a list of dicts, where 
    - each dict corresponds to one layer.
    """
    programs = defaultdict(list)

    l = len(metrics['program_id'])
    assert all(len(v) == l for v in metrics.values())

    for program_id, *values in zip(metrics['program_id'], *metrics.values()):
        programs[program_id].append({
            k: v for k, v in zip(metrics.keys(), values)
        })

    return list(programs.values())


def get_programs_by_length(programs: list[list[dict]]) -> dict[int, float]:
    """Reshuffle to get dict mapping: program_length -> list[programs]"""
    programs_by_length = defaultdict(list)
    for program in programs:
        length = program[0]['n_sops']
        assert all([x['n_sops'] == length for x in program])
        programs_by_length[length].append(program)
    return programs_by_length
    

def compute_overall_accuracy(layers: list[dict], assert_same_program=False) -> float:
    """Get overall accuracy for a list of layers. That is, the
    fraction of tokens that were correctly predicted. 
    Assume layers is a list of dicts, where each dict 
    corresponds to predictions for one layer.
    """
    if assert_same_program:
        assert all([x['program_id'] == layers[0]['program_id'] for x in layers])
    masks = [x['mask'] for x in layers]
    correct_preds = [x['correct_preds'] for x in layers]
    return np.sum(correct_preds) / np.sum(masks)


def compute_all(data, save=False):
    metrics = compute_metrics(data=data)
    metrics.update({
        "program_id": data['program_id'],
        "n_sops": data['n_sops'],
        "rasp_tok": data['rasp_tok'],
    })

    if save:
        savepath = paths.data_dir / "metrics-lib.pkl"
        with open(savepath, 'wb') as f:
            pickle.dump(metrics, f)


    programs = get_program_list(metrics)
    reconstruction_frac = np.mean([compute_overall_accuracy(x) == 1 for x in programs])

    # fraction of all tokens recovered, by length
    accs_by_length = {
        length: compute_overall_accuracy([l for p in programs for l in p])
        for length, programs in get_programs_by_length(programs).items()
    }


    # fraction of programs recovered 100%, by length
    rf_by_length = {
        length: np.mean([compute_overall_accuracy(p, True) == 1 for p in programs])
        for length, programs in get_programs_by_length(programs).items()
    }

    print()
    print("Overall program acc:", reconstruction_frac)
    pprint.pprint("Overall token acc by length:", accs_by_length)
    pprint.pprint("Program acc by length:", rf_by_length)

    return reconstruction_frac, accs_by_length, rf_by_length


for data_name, data in test_datasets.items():
    print()
    print()
    print(data_name)
    compute_all(data, save=(data_name == 'lib'))




def main():
    parser = argparse.ArgumentParser(description='Test a trained model.')

    # wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--checkpoint_path', type=str, default=None)

    args = parser.parse_args()

    if args.checkpoint_path is None:
        raise ValueError("Must specify checkpoint path.")

    args.tags.append("HPC" if on_cluster else "local")
    if args.wandb_run_name is not None:
        args.wandb_run_name += str(int(time()))
    logger.info("Args:\n%s", pprint.pformat(vars(args)))
    rng = jax.random.PRNGKey(args.seed)
    np_rng = np.random.default_rng()
    metrics_logger = Logger()

    model, params = get_model(args.checkpoint_path)
    test_datasets, seq_len = get_data(model.config.emb_dim)
    log_stuff(test_datasets, params, model)
    loss_fn = create_loss_fn(model.apply)

    
    print()
    print()
    for x in test_datasets.values():
        print("dataset keys:", x.keys())
    print()
    print()


    def compute_metrics(data, bs=128):
        """One forward pass over a test set."""
        out = dict(logits=[], mask=[], correct_preds=[])
        dummy_rng = jax.random.PRNGKey(0)  # dropout is off
        for batch in data_iterator(data, bs, stacked_tree=True):
            chex.assert_shape(batch["rasp_tok"], (None, MAX_RASP_LENGTH)) 
            chex.assert_shape(batch["weights"],
                              (None, MAX_WEIGHTS_LENGTH/model.config.emb_dim, model.config.emb_dim))
            
            loss, aux = loss_fn(params, dummy_rng, batch, is_training=False)

            for k in out.keys():
                out[k].append(aux[k])

        out = {k: np.concatenate(v) for k, v in out.items()}
        chex.assert_equal_shape_prefix(list(out.values()) + [data['rasp_tok']], prefix_len=2)
        return out


    for name, data in test_datasets.items():
        m = compute_metrics(data)
        fracs_corr = get_fracs_correct_by_program(
            program_ids=m['program_id'],
            correct_preds=m['correct_preds'],
            mask=m['mask'],
        )

        print()
        print(name)
        print("Program accuracy:", np.mean(fracs_corr))

        if name == 'lib':
            savepath = paths.data_dir / "metrics-lib.pkl"
            with open(savepath, 'wb') as f:
                pickle.dump(m, f)


        plt.hist(fracs_corr, bins=20)



if __name__ == "__main__":
    main()