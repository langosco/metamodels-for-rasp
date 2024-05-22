import os
from time import time
from datetime import datetime
import json
from tqdm import tqdm
from pathlib import Path

import jax
import chex
import numpy as np
import orbax.checkpoint
import optax

from metamodels_for_rasp import interactive
from metamodels_for_rasp import output_dir
from metamodels_for_rasp.train import Logger
from metamodels_for_rasp.logger_config import setup_logger, setup_data_logger
from metamodels_for_rasp.utils import compute_fracs_correct_by_program
from metamodels_for_rasp.experiments import common

from decompile_tracr.dataset import dataloading
from decompile_tracr.dataset.config import load_config


#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_debug_nans", True)


logger = setup_logger(__name__)
data_logger = setup_data_logger(logfile="test.log")
START_TIME = time()

args = common.parse_args()
rng = jax.random.key(args.seed)
np_rng = np.random.default_rng()

train_loader, test_loader = common.get_dataloaders(
    args, np_rng, groups=["train", "test"])

subrng, rng = jax.random.split(rng)
model, updater, state, _ = common.init(subrng, args, train_loader)
metrics_logger = Logger()

if args.use_wandb:
    raise NotImplementedError("Wandb not supported for testing.")


common.log_metadata(args, model, dict(test=test_loader), state, None)


def compute_metrics(state, dataloader, name="test"):
    out = dict(
        mask=[], 
        correct_preds=[], 
        program_id=[], 
        program_length=[],
    )

    data_logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - "
                        f"Logging step {state.step}. Data: {name}.")
    
    metrics_list = []
    for i, batch in enumerate(dataloader):
        chex.assert_shape(batch["tokens"], (None, None)) 
        chex.assert_shape(batch["weights"],
                            (None, None, None))

        state, metrics, aux = updater.compute_val_metrics(
            state, batch, name=name)
        metrics_list.append(metrics)

        for k in out.keys():
            if k in aux.keys():
                out[k].append(aux[k])

        out["program_length"].append(batch["n_sops"])
    
        if i < 10:
            data_logger.info(f"Step {state.step}, {name} batch {i}:")
            for idx in range(20):
                try:
                    data_logger.info(f"Example {idx}:")
                    common.log_rasp_snippet(
                        data_logger=data_logger,
                        tokens=batch['tokens'][idx],
                        preds=aux['preds'][idx],
                        end=100,
                    )
                except IndexError:
                    break
        
    data_logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - "
                        f"Done logging snippets.")
    data_logger.info("\n=======================================\n")

    out = {k: np.concatenate(v) for k, v in out.items()}

    reconstruction_fracs = compute_fracs_correct_by_program(
        program_ids=out['program_id'],
        correct_preds=out['correct_preds'],
        mask=out['mask'],
    )
    reconstruction_fracs = np.array(reconstruction_fracs)

    metrics.update({
        f"{name}/program_accuracy_75": np.mean(reconstruction_fracs > 0.8),
        f"{name}/program_accuracy_90": np.mean(reconstruction_fracs > 0.9),
        f"{name}/program_accuracy_95": np.mean(reconstruction_fracs > 0.95),
        f"{name}/program_accuracy_98": np.mean(reconstruction_fracs > 0.98),
        f"{name}/program_accuracy": np.mean(reconstruction_fracs == 1.0),
        f"{name}/program_frac_correct": np.mean(reconstruction_fracs),
        f"{name}/program_frac_correct_std": np.std(reconstruction_fracs),
    })


    # split by program length
    metrics_by_program_length = {}
    for l in range(1, 11):
        length_l_mask = out["program_length"] == l
        if not length_l_mask.any():
            continue

        length_l_outputs = {
            k: v[length_l_mask] for k, v in out.items()
        }
        length_l_rec_fracs = np.array(compute_fracs_correct_by_program(
            program_ids=length_l_outputs['program_id'],
            correct_preds=length_l_outputs['correct_preds'],
            mask=length_l_outputs['mask'],
        ))
        metrics_by_program_length[l] = {
            f"{name}/program_accuracy_75": np.mean(length_l_rec_fracs > 0.8),
            f"{name}/program_accuracy_90": np.mean(length_l_rec_fracs > 0.9),
            f"{name}/program_accuracy_95": np.mean(length_l_rec_fracs > 0.95),
            f"{name}/program_accuracy_98": np.mean(length_l_rec_fracs > 0.98),
            f"{name}/program_accuracy": np.mean(length_l_rec_fracs == 1.0),
            f"{name}/program_frac_correct": np.mean(length_l_rec_fracs),
            f"{name}/program_frac_correct_std": np.std(length_l_rec_fracs),
            f"{name}/accuracy": (
                out['correct_preds'].sum() / out['mask'].sum())
        }


    return metrics, metrics_by_program_length


start = time()
logger.info("Start testing.")
metrics, metrics_by_length = compute_metrics(state, test_loader, name="test")

print()
print()
print()

logger.info("Test metrics:")
logger.info("==============")
for k, v in metrics.items():
    logger.info(f"{k}: {v:.4f}")

print()
print()
logger.info("Metrics by program length:")
logger.info("==============")
for length, m in metrics_by_length.items():
    logger.info(f"Length {length}:")
    for k, v in m.items():
        logger.info(f"{k}: {v:.4f}")
    print()


# lib
lib_dataset_path = args.data_config.paths.dataset
dataloading.DataLoader(
    loadfile=lib_dataset_path,
    group="lib",
    batch_size=4,
    process_fn=test_loader.process_fn,
    max_datapoints=100,
)

lib_metrics, lib_metrics_by_length = compute_metrics(state, test_loader, name="lib")

print()
print()
print()

logger.info("Metrics on handcrafted examples:")
logger.info("==============")
for k, v in lib_metrics.items():
    logger.info(f"{k}: {v:.4f}")

print()
print()


def to_float_recursive(d: dict):
    for k, v in d.items():
        try:
            d[k] = v.item()
        except AttributeError:
            # assume v is dict-like
            d[k] = to_float_recursive(v)
    return d


# save
data = dict(
    test=metrics,
    by_length=metrics_by_length,
    lib=lib_metrics,
)
data = to_float_recursive(data)
checkpoint_name = args.restore_checkpoint_from
savefile = Path(output_dir) / "results" / f"{checkpoint_name}.json"
savefile.parent.mkdir(parents=True, exist_ok=True)

logger.info(f"Saving results to {savefile}")
with open(savefile, "w") as f:
    json.dump(data, f, indent=4)