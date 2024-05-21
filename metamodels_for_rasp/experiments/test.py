import os
from time import time
from datetime import datetime
import json
from tqdm import tqdm

import jax
import chex
import numpy as np
import orbax.checkpoint
import optax

from metamodels_for_rasp import interactive
from metamodels_for_rasp.train import Logger
from metamodels_for_rasp.logger_config import setup_logger, setup_data_logger
from metamodels_for_rasp.utils import compute_fracs_correct_by_program
from metamodels_for_rasp.experiments import common


#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_debug_nans", True)


logger = setup_logger(__name__)
data_logger = setup_data_logger(logfile="test.log")
START_TIME = time()

args = common.parse_args()
rng = jax.random.key(args.seed)
np_rng = np.random.default_rng()

train_loader, _, test_loader = common.get_dataloaders(args, np_rng)

subrng, rng = jax.random.split(rng)
model, updater, state, _ = common.init(subrng, args, train_loader)
metrics_logger = Logger()

if args.use_wandb:
    raise NotImplementedError("Wandb not supported for testing.")


common.log_metadata(args, model, dict(test=test_loader), state, None)


def compute_metrics(state, dataloader, name="test"):
    out = dict(mask=[], correct_preds=[], program_id=[])
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
            out[k].append(aux[k])
        out["program_length"] = batch["program_length"]
    
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

    # split by program length
    metrics_by_program_length = {}
    for l in range(1, 11):
        length_l_mask = out["program_length"] == l
        length_l_outputs = {
            k: v[length_l_mask] for k, v in out.items()
        }
        length_l_rec_fracs = compute_fracs_correct_by_program(
            program_ids=length_l_outputs['program_id'],
            correct_preds=length_l_outputs['correct_preds'],
            mask=length_l_outputs['mask'],
        )
        metrics_by_program_length[l] = {
            f"{name}/program_accuracy_75": np.mean(length_l_rec_fracs > 0.8),
            f"{name}/program_accuracy_90": np.mean(length_l_rec_fracs > 0.9),
            f"{name}/program_accuracy_95": np.mean(length_l_rec_fracs > 0.95),
            f"{name}/program_accuracy_98": np.mean(length_l_rec_fracs > 0.98),
            f"{name}/program_accuracy": np.mean(length_l_rec_fracs == 1.0),
            f"{name}/program_frac_correct": np.mean(length_l_rec_fracs),
            f"{name}/program_frac_correct_std": np.std(length_l_rec_fracs),
        }

        metrics_by_program_length[f"{name}/accuracy"] = (
            out['correct_preds'].sum() / out['mask'].sum())



    metrics.update({
        f"{name}/program_accuracy_75": np.mean(reconstruction_fracs > 0.8),
        f"{name}/program_accuracy_90": np.mean(reconstruction_fracs > 0.9),
        f"{name}/program_accuracy_95": np.mean(reconstruction_fracs > 0.95),
        f"{name}/program_accuracy_98": np.mean(reconstruction_fracs > 0.98),
        f"{name}/program_accuracy": np.mean(reconstruction_fracs == 1.0),
        f"{name}/program_frac_correct": np.mean(reconstruction_fracs),
        f"{name}/program_frac_correct_std": np.std(reconstruction_fracs),
    })

    return metrics


start = time()
logger.info("Start testing.")
metrics = compute_metrics(state, test_loader, name="test")

print()
print()
print()

logger.info("Test metrics:")
logger.info("==============")
for k, v in metrics.items():
    logger.info(f"{k}: {v:.4f}")