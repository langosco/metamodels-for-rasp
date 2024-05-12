import os
from time import time
from datetime import datetime
import json
from tqdm import tqdm

import jax
import chex
import numpy as np
import orbax.checkpoint

from metamodels_for_rasp import interactive
from metamodels_for_rasp.train import Logger
from metamodels_for_rasp.logger_config import setup_logger, setup_data_logger
from metamodels_for_rasp.utils import compute_fracs_correct_by_program
from metamodels_for_rasp.experiments import common

from decompile_tracr.dataset import config as dataset_config


#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_debug_nans", True)

os.environ['XLA_FLAGS'] = (
#    '--xla_gpu_enable_triton_softmax_fusion=true '  # results in 'Command terminated by signal 11'
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)


logger = setup_logger(__name__)
data_logger = setup_data_logger(logfile="train.log")
START_TIME = time()
VAL_DATA_RATIO = 0.1
MAX_RASP_LENGTH = dataset_config.MAX_RASP_LENGTH
MAX_WEIGHTS_LENGTH = dataset_config.MAX_WEIGHTS_LENGTH


def main():
    args = common.parse_args()
    rng = jax.random.key(args.seed)
    np_rng = np.random.default_rng()

    train_loader, val_loader, test_loader = common.get_dataloaders(args, np_rng)

    subrng, rng = jax.random.split(rng)
    model, updater, state, run = common.init(subrng, args, train_loader)
    metrics_logger = Logger()

    if args.use_wandb:
        checkpoint_savename = run.name
    else:
        checkpoint_savename = f"run_{int(time())}"

    common.log_metadata(
        args, model, dict(train=train_loader, val=val_loader), state, 
        checkpoint_savename,
    )


    def train_one_epoch(state):
        """Train for one epoch. Return updated state."""
        disable_tqdm = not interactive or args.disable_tqdm
        stop_training = False
        past_steps = state.step
        for i, batch in enumerate(tqdm(
            train_loader,
            disable=disable_tqdm,
            desc=f"Training epoch {train_loader.epoch_count}",
        )):
            step = past_steps + i
            chex.assert_shape(batch["weights"], 
                              (args.bs, None, None))
            chex.assert_shape(batch["tokens"], (args.bs, None)) 

            state, train_metrics = updater.update(state, batch)
            metrics_logger.write(state, train_metrics, name="train")

            if step in [1, 2, 4, 8, 16, 32, 64, 128]:
                # log more frequently early on
                metrics_logger.flush_mean(state, name="train",
                        verbose=disable_tqdm, extra_metrics={"epoch": epoch})

            if time() - start > args.max_runtime * 60:
                logger.info("Maximum runtime reached. Stopping training.")
                stop_training = True
                break

            if step > args.max_steps:
                logger.info("Maximum number of steps reached. Stopping "
                            "training.")
                stop_training = True
                break
            
        if step > 128:
            # log training metrics once per epoch
            metrics_logger.flush_mean(state, name="train",
                    verbose=disable_tqdm, extra_metrics={"epoch": epoch})

        return state, stop_training


    def compute_metrics(state, dataloader, name="test"):
        out = dict(mask=[], correct_preds=[], program_id=[])
        data_logger.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - "
                         f"Logging step {state.step}. Data: {name}.")
        for i, batch in enumerate(dataloader):
            chex.assert_shape(batch["tokens"], (None, None)) 
            chex.assert_shape(batch["weights"],
                              (None, None, None))

            state, val_metrics, aux = updater.compute_val_metrics(
                state, batch, name=name)
            metrics_logger.write(state, val_metrics, name=name)

            for k in out.keys():
                out[k].append(aux[k])
        
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

        program_acc = np.mean(reconstruction_fracs == 1.0)
        program_acc_50 = np.mean(reconstruction_fracs > 0.5)

        metrics_logger.flush_mean(
            state, 
            name=name, 
            verbose=False, 
            extra_metrics={
                "epoch": epoch,
                f"{name}/program_accuracy": program_acc,
                f"{name}/program_accuracy_50": program_acc_50,
                f"{name}/program_accuracy_90": np.mean(reconstruction_fracs > 0.9),
                f"{name}/program_accuracy_95": np.mean(reconstruction_fracs > 0.95),
                f"{name}/program_accuracy_98": np.mean(reconstruction_fracs > 0.98),
                f"{name}/program_frac_correct": np.mean(reconstruction_fracs),
                f"{name}/program_frac_correct_std": np.std(reconstruction_fracs),
            }
        )
            
        return state


    # Training loop
    start = time()
    logger.info("Start training.")
    for epoch in range(args.max_epochs):
        if epoch % args.val_every == 0:
            logger.info(f"Validating pre epoch {epoch}.")
            state = compute_metrics(state, val_loader, name="val")

        state, stop_training = train_one_epoch(state)

        if stop_training:
            logger.info(f"Final validation before stopping training "
                        f"at epoch {epoch}.")
            state = compute_metrics(state, val_loader, name="val")
            break
    
    logger.info("=======================================")
    logger.info("Completed.")
    logger.info(f"Total time elapsed since start: {round(time() - START_TIME)} seconds.")


    # save checkpoint when training is done
    if args.save_checkpoint:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        logger.info("Saving checkpoint...")
        checkpointer.save(
            args.checkpoint_dir / checkpoint_savename, (state.params, model.config))

        model_config = {k: v for k, v in vars(model.config).items() 
                        if not any(k.startswith(p) for p in [
                            "kernel", "bias", "posemb", "dtype"])}
        info = {
            'model_config': model_config,
        }
        with open(args.checkpoint_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        logger.info(f"Checkpoint saved to "
                    f"{args.checkpoint_dir}/{checkpoint_savename}.")


if __name__ == "__main__":
    main()
