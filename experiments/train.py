from time import time
import os
import pprint
import json
from collections import defaultdict
from dataclasses import asdict

import jax
import jax.numpy as jnp
import optax
import chex
import numpy as np
import wandb
import argparse
from etils import epath
import orbax.checkpoint
from etils import epath
from tqdm import tqdm
from jax.typing import ArrayLike

from nn_utils import schedules
from metamodels_for_rasp import on_cluster, output_dir, interactive
from metamodels_for_rasp.train import Updater, Logger
from metamodels_for_rasp.logger_config import setup_logger
from metamodels_for_rasp.model import Transformer, TransformerConfig
from metamodels_for_rasp.utils import count_params, data_iterator

from rasp_tokenizer import paths
from rasp_tokenizer import vocab
from rasp_tokenizer.data_utils import load_data, process_data, split_dict_data
from rasp_tokenizer import MAX_RASP_LENGTH, MAX_WEIGHTS_LENGTH


logger = setup_logger(__name__)
START_TIME = time()
VAL_DATA_RATIO = 0.1


def accuracy(logits, targets, mask) -> (float, jnp.ndarray):
    hits = (logits.argmax(axis=-1) == targets) * mask
    return hits.sum() / mask.sum(), hits


def get_mask(tokens):
    """Get mask for padding tokens."""
    return jnp.where(tokens == vocab.pad_id, 0, 1)


def create_loss_fn(model_forward: callable):
    def loss_fn(
            params: dict,
            rng: ArrayLike,
            batch: dict,
            is_training: bool = True,
    ) -> (float, dict):
        """Compute loss for a batch."""
        tokens = batch['rasp']
        loss_mask = get_mask(tokens)

        outputs = model_forward(
            {"params": params},
            batch,
            is_training=is_training,
            rngs={"dropout": rng},
        )

        logits = outputs[:, -tokens.shape[1]-1:-1, :]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
        loss = jnp.sum(loss * loss_mask) / jnp.sum(loss_mask)
        acc, correct_preds = accuracy(logits, tokens, loss_mask)
        metrics = {"accuracy": acc}
        aux = dict(outputs=outputs, logits=logits, metrics=metrics, 
                   correct_preds=correct_preds, mask=loss_mask)
        return loss, aux
    return loss_fn


def main():
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    # model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.00)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=None)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--augment', action='store_true', help="Augment base models via permutations")

    # init
    parser.add_argument('--in_factor', type=float, default=1.0, help="muP scale factor for input")
    parser.add_argument('--out_factor', type=float, default=1.0, help="muP scale factor for output")
    parser.add_argument('--attn_factor', type=float, default=1.0, help="muP scale factor for attention")
    parser.add_argument('--init_scale', type=float, default=1.0)

    # training
    parser.add_argument('--nsteps', type=int, default=100_000)
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

    args = parser.parse_args()

    args.tags.append("HPC" if on_cluster else "local")
    logger.info("Args:\n%s", pprint.pformat(vars(args)))
    rng = jax.random.PRNGKey(args.seed)

    # Data
    data, test_data = load_data()
    if len(data) < args.ndata:
        logger.warning(f"Requested {args.ndata} datapoints, but only "
                       f"{len(data)} available.")
    else:
        data = list(data)[:args.ndata]
    data = process_data(
        data=data, 
        d_model=args.d_model, 
        max_rasp_len=MAX_RASP_LENGTH,
        max_weights_len=MAX_WEIGHTS_LENGTH,
    )

    train_data, val_data = split_dict_data(
        data, val_ratio=VAL_DATA_RATIO)
    
    # normalize weights
    w_mean, w_std = train_data["weights"].mean(), train_data["weights"].std()
    train_data['weights'] = (train_data['weights'] - w_mean) / w_std
    val_data['weights'] = (val_data['weights'] - w_mean) / w_std

    # clip again
#    train_data["weights"] = np.clip(train_data["weights"], -1, 1)
#    val_data["weights"] = np.clip(val_data["weights"], -1, 1)


    args.nsteps = min(args.nsteps, args.max_epochs * args.ndata // args.bs)
    seq_len=MAX_RASP_LENGTH + MAX_WEIGHTS_LENGTH/args.d_model
    assert seq_len.is_integer()
    seq_len = int(seq_len)

    config = TransformerConfig(
        vocab_size=vocab.size,
        output_vocab_size=vocab.size,
        emb_dim=args.d_model,
        num_heads=max(1, int(args.d_model / 64)),
        num_layers=args.num_layers if args.num_layers is not None else int(max(args.d_model / 42, 1)),
        qkv_dim=args.d_model,
        mlp_dim=args.d_model * 4,
        max_len=seq_len,
        dropout_rate=args.dropout_rate,
        attention_dropout_rate=args.dropout_rate,
    )
    model = Transformer(config=config)


    model_scale = args.d_model / 1024
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
    opt = optimizer(lr=schedule, wd=args.wd, clip_value=1.)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    metrics_logger = Logger()

    dummy_batch = {
        "weights": data['weights'][:1],
        "rasp": data['rasp'][:1],
    }
    subrng, rng = jax.random.split(rng)
    state = updater.init_train_state(subrng, dummy_batch)


    checkpoint_savedir = epath.Path(output_dir) / "mm-checkpoints" \
        / "decompile"
    checkpoint_savename = f"run_{int(time())}"

    wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project=f"inverse-tracr",
        tags=args.tags,
        notes=args.notes,
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

    logger.info(f"Tags: {args.tags}")
    logger.info("Number of training examples: "
                f"{len(train_data['rasp'])}.")
    logger.info("Number of validation examples: "
                f"{len(val_data['rasp'])}.")
    logger.info(f"Total number of steps: {args.nsteps}")
    logger.info(f"Number of parameters in meta-model: "
                f"{count_params(state.params) / 1e6} Million")
    logger.info(f"Data shapes: weights: {train_data['weights'].shape}, "
                f"rasp: {train_data['rasp'].shape}")
    logger.info(f"Train data (weights) mean: {train_data['weights'].mean()}, "
                f"std: {train_data['weights'].std()}")
    logger.info(f"Percent of weights zero: "
                f"{round(100 * (train_data['weights'] == 0).sum() / train_data['weights'].size, 2)}%")
    logger.info(f"Expected sequence length: {seq_len}.")
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
            chex.assert_shape(batch["rasp"], (args.bs, MAX_RASP_LENGTH)) 
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

        metrics_logger.flush_mean(state, name="train",
                verbose=disable_tqdm, extra_metrics={"epoch": epoch})
        return state, stop_training

    
    def get_program_accuracy(program_ids: list, correct_preds: list):
        """Get accuracy per program."""
        preds_by_prog = defaultdict(list)
        for prog_id, cp in zip(program_ids, correct_preds):
            preds_by_prog[prog_id].append(cp)
        
        full_program_correct = [
            all(preds) for preds in preds_by_prog.values()
        ]
        return np.mean(full_program_correct)


    def validate(state):
        program_ids = []
        correct_preds = []
        for batch in data_iterator(val_data, args.bs, stacked_tree=True):
            chex.assert_shape(batch["rasp"], (None, MAX_RASP_LENGTH)) 
            chex.assert_shape(batch["weights"],
                              (None, MAX_WEIGHTS_LENGTH/args.d_model, args.d_model))

            state, val_metrics, aux = updater.compute_val_metrics(
                state, batch)
            metrics_logger.write(state, val_metrics, name="val")

            mask = np.array(aux['mask'], dtype=bool)
            program_ids += batch['program_id'].flatten().tolist()
            correct_preds += aux['correct_preds'][mask].flatten().tolist()

        metrics_logger.flush_mean(
            state, 
            name="val", 
            verbose=disable_tqdm, 
            extra_metrics={
                "epoch": epoch,
                "program_accuracy": get_program_accuracy(program_ids, correct_preds),
            }
        )
            
        return state


    # Training loop
    start = time()
    disable_tqdm = not interactive or args.disable_tqdm
    for epoch in range(args.max_epochs):
        if epoch % args.val_every == 0:
            state = validate(state)

        state, stop_training = train(state)

        if stop_training:
            validate(state)
            break
    
    logger.info("=======================================")
    logger.info("Completed.")
    logger.info(f"Total time elapsed since start: {round(time() - START_TIME)} seconds.")


    # save checkpoint when training is done
    if args.save_checkpoint:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        logger.info("Saving checkpoint...")
        checkpointer.save(
            checkpoint_savedir / checkpoint_savename, state.params)

        model_config = {k: v for k, v in vars(model).items() 
                        if not k.startswith("_")}
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