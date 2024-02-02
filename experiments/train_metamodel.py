from time import time
import os
import pprint
import json
from collections import defaultdict
from dataclasses import asdict
import argparse
from tqdm import tqdm

import jax
import optax
import chex
import numpy as np
import wandb
import orbax.checkpoint
from jax.typing import ArrayLike
from etils import epath

from nn_utils import schedules
from metamodels_for_rasp import on_cluster, output_dir, interactive
from metamodels_for_rasp.train import Updater, Logger
from metamodels_for_rasp.logger_config import setup_logger, setup_data_logger
from metamodels_for_rasp.model import Transformer, TransformerConfig
from metamodels_for_rasp.utils import count_params, data_iterator, color_sequence, create_loss_fn, compute_fracs_correct_by_program

from rasp_tokenizer import paths
from rasp_tokenizer import vocab
from rasp_tokenizer.data_utils import load_and_process_data, split_dict_data
from rasp_tokenizer import MAX_RASP_LENGTH, MAX_WEIGHTS_LENGTH
from rasp_tokenizer import tokenizer


#jax.config.update("jax_disable_jit", True)


logger = setup_logger(__name__)
data_logger = setup_data_logger()
START_TIME = time()
VAL_DATA_RATIO = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--seed', type=float, default=42, help='random seed')

    # model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=None)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--augment', action='store_true', help="Augment base models via permutations")

    # init
    parser.add_argument('--in_factor', type=float, default=1.0, help="muP scale factor for input")
    parser.add_argument('--out_factor', type=float, default=1.0, help="muP scale factor for output")
    parser.add_argument('--attn_factor', type=float, default=1.0, help="muP scale factor for attention")
    parser.add_argument('--init_scale', type=float, default=1.0)

    # training
    parser.add_argument('--nsteps', type=int, default=np.inf)
    parser.add_argument('--max_runtime', type=int, help='Max runtime in minutes', default=np.inf)
    parser.add_argument('--max_epochs', type=int, default=10**8)
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')
    parser.add_argument('--val_every', type=int, default=5)

    # adam
    parser.add_argument('--lr', type=float, help='Learning rate', 
                        default=1e-4)
    parser.add_argument('--wd', type=float, help='Weight decay', 
                        default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=64)
    parser.add_argument('--adam_b1', type=float, default=0.1)
    parser.add_argument('--adam_b2', type=float, default=0.001)
    parser.add_argument('--adam_eps', type=float, default=1e-8)

    # data
    parser.add_argument('--ndata', type=int, 
                        help='Number of training datapoints', default=20_000)
    parser.add_argument('--data_dir', type=str, default=paths.data_dir)

    # wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    args.tags.append("HPC" if on_cluster else "local")

    if args.wandb_run_name is not None:
        args.wandb_run_name += str(int(time()))

    args.nsteps = min(args.nsteps, args.max_epochs * args.ndata // args.bs)

    logger.info("Args:\n%s", pprint.pformat(vars(args)))
    logger.info("\n")
    data_logger.info("Args:\n%s", pprint.pformat(vars(args)))
    data_logger.info("\n")

    return args



def load_data(args, np_rng) -> (list, list, list):
    data = load_and_process_data(
        rng=np_rng,
        ndata=args.ndata,
        shuffle=True,
        name="train",
        d_model=args.d_model,
        max_rasp_len=MAX_RASP_LENGTH,
        max_weights_len=MAX_WEIGHTS_LENGTH,
    )

    train_data, val_data = split_dict_data(
        data, val_ratio=VAL_DATA_RATIO)

    test_datasets = {
        name: load_and_process_data(
            rng=None,
            ndata=None,
            shuffle=False,
            name=name,
            d_model=args.d_model,
            max_rasp_len=MAX_RASP_LENGTH,
            max_weights_len=MAX_WEIGHTS_LENGTH,
        ) for name in ["lib", "test"]
    }

#    # normalize weights
    w_std = train_data["weights"].std()
    logger.info(f"Weight std before normalization: {w_std}")
    train_data['weights'] = train_data['weights'] / w_std
    val_data['weights'] = val_data['weights'] / w_std

    for v in test_datasets.values():
        v['weights'] = v['weights'] / w_std

    return train_data, val_data, test_datasets


def get_model(args):
    weight_len = MAX_WEIGHTS_LENGTH/args.d_model
    assert weight_len.is_integer()
    weight_len = int(weight_len)
    seq_len = MAX_RASP_LENGTH + weight_len

    config = TransformerConfig(
        weight_len=weight_len,
        rasp_tok_len=MAX_RASP_LENGTH,
        vocab_size=vocab.size,
        output_vocab_size=vocab.size,
        emb_dim=args.d_model,
        num_heads=max(1, int(args.d_model / 64)),
        num_layers=args.num_layers if args.num_layers is not None else int(max(args.d_model / 42, 1)),
        qkv_dim=args.d_model,
        mlp_dim=args.d_model * 4,
        max_len=seq_len,
        dropout_rate=0.0,  # no dropout on the residual stream
        attention_dropout_rate=args.dropout_rate,
#        posemb_init=nn.initializers.zeros,
    )
    return Transformer(config=config)


def init_updater(rng, args, model, train_data):
    """Initialize the model and set up the optimizer."""
    @optax.inject_hyperparams
    def optimizer(lr: float, wd: float, clip_value: float) -> optax.GradientTransformation:
        opt = optax.adamw(
            learning_rate=lr,
            b1=1-args.adam_b1,
            b2=1-args.adam_b2,
            eps=args.adam_eps,
            weight_decay=wd,
        )
        return optax.chain(
            optax.clip_by_global_norm(clip_value),
            opt,
        )

    schedule = schedules.constant_with_warmup_and_cooldown(
        args.lr,
        args.nsteps, 
        warmup_length=args.nsteps//5, 
        cooldown_start=int(args.nsteps*0.75), 
        max_lr=args.lr*4,
    )
    opt = optimizer(lr=schedule, wd=args.wd, clip_value=20.)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)

    # init
    dummy_batch = {
        "weights": train_data['weights'][:1],
        "rasp_tok": train_data['rasp_tok'][:1],
    }
    subrng, rng = jax.random.split(rng)
    state = updater.init_train_state(subrng, dummy_batch)
    return updater, state


def main():
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)
    np_rng = np.random.default_rng()

    with jax.default_device(jax.devices("cpu")[0]):
        train_data, val_data, test_datasets = load_data(args, np_rng)

    model = get_model(args)
    subrng, rng = jax.random.split(rng)
    batch_for_init = {
        "weights": train_data['weights'][:1],
        "rasp_tok": train_data['rasp_tok'][:1],
    }
    updater, state = init_updater(subrng, args, model, batch_for_init)
    metrics_logger = Logger()


    run = wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project=f"inverse-tracr",
        tags=args.tags,
        notes=args.notes,
        name=args.wandb_run_name,
        config={
            "lr": args.lr,
            "weight_decay": args.wd,
            "batchsize": args.bs,
            "max_epochs": args.max_epochs,
            "model_config": asdict(model),
            "num_datapoints": len(train_data),
            "adam/b1": args.adam_b1,
            "adam/b2": args.adam_b2,
            "adam/eps": args.adam_eps,
            "slurm_job_id": os.environ.get('SLURM_JOB_ID'),
            "slurm_job_name": os.environ.get('SLURM_JOB_NAME'),
            "save_checkpoint": args.save_checkpoint,
        },
    )

    checkpoint_savedir = epath.Path(output_dir) / "mm-checkpoints" \
        / "decompile"
    if args.use_wandb:
        checkpoint_savename = run.name
    else:
        checkpoint_savename = f"run_{int(time())}"


    logger.info(f"Tags: {args.tags}")
    logger.info("Number of training examples: "
                f"{len(train_data['rasp_tok'])}.")
    logger.info("Number of validation examples: "
                f"{len(val_data['rasp_tok'])}.")
    for test_data in test_datasets.values():
        logger.info("Number of test examples: "
                    f"{len(test_data['rasp_tok'])}.")
    logger.info(f"Total number of steps: {args.nsteps}")
    logger.info(f"Number of parameters in meta-model: "
                f"{count_params(state.params) / 1e6} Million")
    logger.info(f"Data shapes: weights: {train_data['weights'].shape}, "
                f"rasp_tok: {train_data['rasp_tok'].shape}")
    logger.info(f"Train data (weights) mean: {train_data['weights'].mean()}, "
                f"std: {train_data['weights'].std()}")
    for test_data in test_datasets.values():
        logger.info(f"Test data (weights) mean: {test_data['weights'].mean()}, "
                    f"std: {test_data['weights'].std()}")
    logger.info(f"Percent of weights zero: "
                f"{round(100 * (train_data['weights'] == 0).sum() / train_data['weights'].size, 2)}%")
    logger.info(f"Expected sequence length: {model.config.max_len}.")
    if args.save_checkpoint:
        logger.info(f"Saving final checkpoint to {checkpoint_savedir}/{checkpoint_savename}.")


    def train(state):
        # TODO: shuffle train data
        stop_training = False
        for batch in tqdm(
                data_iterator(train_data, args.bs, stacked_tree=True, skip_last=True),
                disable=disable_tqdm,
                desc="Training",
            ):
            chex.assert_shape(batch["weights"], 
                              (args.bs, MAX_WEIGHTS_LENGTH/args.d_model, args.d_model))
            chex.assert_shape(batch["rasp_tok"], (args.bs, MAX_RASP_LENGTH)) 
            state, train_metrics = updater.update(state, batch)
            metrics_logger.write(state, train_metrics, name="train")

            if time() - start > args.max_runtime * 60:
                logger.info("Maximum runtime reached. Stopping training.")
                stop_training = True
                break

            if state.step > args.nsteps:
                logger.info("Maximum number of steps reached. Stopping "
                            "training.")
                stop_training = True
                break
            
            if state.step in [1, 2, 4, 8, 16, 32, 64, 128]:
                # log more frequently early on
                metrics_logger.flush_mean(state, name="train",
                        verbose=disable_tqdm, extra_metrics={"epoch": epoch})

        if state.step > 128:
            # log once per epoch
            metrics_logger.flush_mean(state, name="train",
                    verbose=disable_tqdm, extra_metrics={"epoch": epoch})
        return state, stop_training


    def log_rasp_snippet(
            tokens: ArrayLike, 
            preds: ArrayLike, 
            name: str,
            snip_at: int = 10,
        ):
        """Args:
            - tokens: 1D array of ground-truth tokens
            - preds: 1D array of predictions
            - name: 'train', 'val', or 'test'
            - snip_at: number of tokens to include in snippet
        """
        correct_preds = (tokens == preds)[:snip_at]
        rasp_snippet = tokenizer.decode(tokens[:snip_at])
        decoded_preds = tokenizer.decode(preds[:snip_at])
        try:
            eos_idx = rasp_snippet.index("EOS") + 1
        except ValueError:
            eos_idx = len(decoded_preds)

        correct_preds = correct_preds[:eos_idx]
        rasp_snippet = rasp_snippet[:eos_idx]

        rasp_snippet = color_sequence(rasp_snippet, correct_preds)
        data_logger.info(f"{name}: {rasp_snippet} (true)")

        decoded_preds = decoded_preds[:eos_idx]
        data_logger.info(f"{name}: {decoded_preds} (preds)")


    def compute_metrics(state, data, name="test"):
        out = dict(mask=[], correct_preds=[], program_id=[])
        for i, batch in enumerate(
                    data_iterator(data, args.bs, stacked_tree=True)):

            chex.assert_shape(batch["rasp_tok"], (None, MAX_RASP_LENGTH)) 
            chex.assert_shape(batch["weights"],
                              (None, MAX_WEIGHTS_LENGTH/args.d_model, args.d_model))

            state, val_metrics, aux = updater.compute_val_metrics(
                state, batch, name=name)
            metrics_logger.write(state, val_metrics, name=name)

            mask = np.array(aux['mask'], dtype=bool)

            for k in out.keys():
                out[k].append(aux[k])
        
            if i < 3:
                for idx in range(10):
                    log_rasp_snippet(
                        tokens=batch['rasp_tok'][idx],
                        preds=aux['preds'][idx],
                        name=name,
                        snip_at=15,
                    )
        data_logger.info("\n")

        out = {k: np.concatenate(v) for k, v in out.items()}

        reconstruction_fracs = compute_fracs_correct_by_program(
            program_ids=out['program_id'],
            correct_preds=out['correct_preds'],
            mask=out['mask'],
        )

        program_acc = np.mean(reconstruction_fracs == 1.0)
        program_acc_50 = np.mean(np.array(reconstruction_fracs) > 0.5)

        metrics_logger.flush_mean(
            state, 
            name=name, 
            verbose=disable_tqdm, 
            extra_metrics={
                "epoch": epoch,
                f"{name}/program_accuracy": program_acc,
                f"{name}/program_accuracy_50": program_acc_50,
                f"{name}/program_accuracy_90": np.mean(np.array(reconstruction_fracs) > 0.9),
            }
        )
            
        return state


    # Training loop
    start = time()
    disable_tqdm = not interactive or args.disable_tqdm
    for epoch in range(args.max_epochs):
        if epoch % args.val_every == 0:
            state = compute_metrics(state, val_data, name="val")
            for name, test_data in test_datasets.items():
                state = compute_metrics(state, test_data, name=name)

        state, stop_training = train(state)

        if stop_training:
            state = compute_metrics(state, val_data, name="val")
            for name, test_data in test_datasets.items():
                state = compute_metrics(state, test_data, name=name)
            break
    
    logger.info("=======================================")
    logger.info("Completed.")
    logger.info(f"Total time elapsed since start: {round(time() - START_TIME)} seconds.")


    # save checkpoint when training is done
    if args.save_checkpoint:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        logger.info("Saving checkpoint...")
        checkpointer.save(
            checkpoint_savedir / checkpoint_savename, (state.params, model.config))

        model_config = {k: v for k, v in vars(model.config).items() 
                        if not any(k.startswith(p) for p in [
                            "kernel", "bias", "posemb", "dtype"])}
        info = {
            'model_config': model_config,
            'ndata': args.ndata,
            'nsteps': args.nsteps,
        }
        with open(checkpoint_savedir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        logger.info(f"Checkpoint saved to "
                    f"{checkpoint_savedir}/{checkpoint_savename}.")


if __name__ == "__main__":
    main()