import os
import h5py
from dataclasses import asdict
import argparse
from pathlib import Path
import time
import logging
import pprint
from typing import Optional

import numpy as np
import einops
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax
import orbax.checkpoint
import wandb

from nn_utils import schedules
from metamodels_for_rasp import on_cluster, output_dir
from metamodels_for_rasp.train import Updater
from metamodels_for_rasp.logger_config import setup_logger
from metamodels_for_rasp.model import Transformer, TransformerConfig
from metamodels_for_rasp.utils import color_sequence, count_params
from rasp_gen.tokenize import vocab
from rasp_gen.tokenize import tokenizer
from rasp_gen.dataset import dataloading
from rasp_gen.dataset.config import load_config


logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--seed', type=float, default=42, help='random seed')

    # model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--num_layers', type=int, help='Number of layers', 
                        default=None)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--restore_checkpoint_from', type=str, default=None,
                        help='Checkpoint name to restore from')
    parser.add_argument('--checkpoint_dir', type=str, 
        default=Path(output_dir) / "mm-checkpoints" / "decompile")

    # training
    parser.add_argument('--max_steps', type=int, default=np.inf)
    parser.add_argument('--max_runtime', type=int, help='Max runtime in minutes', default=np.inf)
    parser.add_argument('--max_epochs', type=int, default=10**5)
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--test', action='store_true',
                        help='Run test mode (no training)')

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
    parser.add_argument('--ndata', type=int, default=np.inf,
                        help='Number of training datapoints (programs)')
    parser.add_argument('--split_layers', action='store_true',
                        help='Load individual layers of base models.')
    parser.add_argument('--symlog', action='store_true')
    parser.add_argument('--data_config', type=str, default=None)
    parser.add_argument('--dummy_data_for_baseline', action='store_true',
                        help='Train using random weights')

    # wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()

    args.tags.append("HPC" if on_cluster else "local")
    if args.restore_checkpoint_from is not None:
        args.tags.append("continued")

    if args.wandb_run_name is not None:
        args.wandb_run_name += str(int(time.time()))
    
    args.data_config = load_config(args.data_config)
    return args


def get_dataloaders(
    args,
    rng: np.random.Generator,
    groups=["train", "val", "test"],
) -> tuple[dict, dict, dict]:
    config = args.data_config
    with h5py.File(config.paths.dataset, "r") as f:
        for_stats = {k: v[:5000] for k, v in f["train"].items()}

    if not len(for_stats['weights']) == 5000:
        raise ValueError("Expected at least 5000 elements in training dataset, "
                         f"but got only {len(for_stats['weights'])}.")

    assert for_stats['weights'].shape[1] == config.max_weights_length, (
        f"Expected shape (n, {config.max_weights_length}) based on config, "
        f"instead got {for_stats['weights'].shape}."
    )

    # TODO: shuffle data? would need to implement in dataloading.py
    if args.symlog:
        for_stats['weights'] = dataloading.symlog(for_stats['weights'])
    w_mean, w_std = (for_stats['weights'].mean(dtype=np.float64), 
                     for_stats['weights'].std(dtype=np.float64))
    logger.info("Weight mean and std before normalization: "
                f"{w_mean}, {w_std}")
    
    @jax.jit
    def process_batch(data):
        data['program_id'] = data['batch_id'] + np.arange(len(data['tokens']))
        if args.symlog:
            data['weights'] = dataloading.symlog(data['weights'])
        data['weights'] = (data['weights'] - w_mean) / w_std
        data['weights'] = einops.rearrange(
            data['weights'], 'b (s d) -> b s d', d=args.d_model)
        if args.dummy_data_for_baseline:
            data['weights'] = jax.random.normal(
                jax.random.key(0), shape=data['weights'].shape)
        return data
    
    return (dataloading.DataLoader(
        loadfile=config.paths.dataset,
        group=group,
        batch_size=args.bs,
        process_fn=process_batch,
        max_datapoints=args.ndata,
    ) for group in groups)


def init(
    rng,
    args,
    dataloader,
    config_updates: Optional[dict] = None,
) -> tuple[Transformer, Updater, dict, "Run"]:
    """Initialize model, optimizer, loss function, and wandb run.
    """
    if args.restore_checkpoint_from is None:
        model = _get_model(args)
        restored_params = None
    else:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        restored_params, model_config = checkpointer.restore(
            args.checkpoint_dir / args.restore_checkpoint_from)
        model_config = {k: v for k, v in model_config.items() if v is not None}
        for k, v in model_config.items():
            try:
                model_config[k] = v.item()
            except AttributeError:
                pass
        if config_updates is not None:
            model_config.update(config_updates)
        model = Transformer(config=TransformerConfig(**model_config))
        logger.info(f"Restored params from {args.restore_checkpoint_from}."
                    " This overrides model config args.")

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
    
    num_steps = min(len(dataloader) * args.max_epochs, args.max_steps)

    schedule = schedules.constant_with_warmup_and_cooldown(
        args.lr,
        total_steps=num_steps, 
        warmup_length=min(num_steps//5, 15_000), 
        cooldown_start=int(num_steps*0.9), 
        max_lr=args.lr*4,
    )
    opt = optimizer(lr=schedule, wd=args.wd, clip_value=20.)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)

    # init
    subrng, rng = jax.random.split(rng)
    dummy_batch = {
        "weights": np.ones((1, *dataloader.shape['weights'][1:])),
        "tokens": np.ones((1, *dataloader.shape['tokens'][1:])),
    }
    state = updater.init_train_state(subrng, dummy_batch, 
                                init_params=restored_params)

    # wandb 
    run = wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project=f"decompile-tracr",
        tags=args.tags,
        notes=args.notes,
        name=args.wandb_run_name,
        config={
            "lr": args.lr,
            "weight_decay": args.wd,
            "batchsize": args.bs,
            "max_epochs": args.max_epochs,
            "model_config": asdict(model),
            "num_datapoints": dataloader.shape['weights'][0],
            "adam/b1": args.adam_b1,
            "adam/b2": args.adam_b2,
            "adam/eps": args.adam_eps,
            "slurm_job_id": os.environ.get('SLURM_JOB_ID'),
            "slurm_job_name": os.environ.get('SLURM_JOB_NAME'),
            "save_checkpoint": args.save_checkpoint,
            "dataset": args.data_config.name,
        },
    )
    return model, updater, state, run


def _get_model(args):
    weight_len = args.data_config.max_weights_length / args.d_model

    assert weight_len.is_integer()
    weight_len = int(weight_len)
    seq_len = args.data_config.max_rasp_length + weight_len

    config = TransformerConfig(
        weight_len=weight_len,
        rasp_tok_len=args.data_config.max_rasp_length,
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
        decode=args.test,
    )
    return Transformer(config=config)


def accuracy(logits, targets, mask) -> tuple[float, jnp.ndarray]:
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
    ) -> tuple[float, dict]:
        """Compute loss for a batch."""
        tokens = batch['tokens']

        outputs = model_forward(
            {"params": params},
            batch,
            is_training=is_training,
            rngs={"dropout": rng},
        )

        loss_mask = get_mask(tokens)
        logits = outputs[:, -tokens.shape[1]-1:-1, :]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
        loss = jnp.sum(loss * loss_mask) / jnp.sum(loss_mask)
        acc, correct_preds = accuracy(logits, tokens, loss_mask)
        metrics = {"accuracy": acc}
        aux = dict(outputs=outputs, logits=logits, metrics=metrics, 
                   correct_preds=correct_preds, mask=loss_mask,
                   preds=logits.argmax(axis=-1), program_id=batch['program_id'])
        return loss, aux
    return loss_fn


def log_rasp_snippet(
    data_logger: logging.Logger,
    tokens: ArrayLike, 
    preds: ArrayLike, 
    start: int = 0,
    end: int = 25,
) -> None:
    """Args:
        - tokens: 1D array of ground-truth tokens
        - preds: 1D array of predictions
        - name: 'train', 'val', or 'test'
        - start: start token index
        - end: end token index
    """
    pad_start = np.where(tokens == vocab.pad_id)[0][0]
    end = min(end, pad_start)  # cut off at first pad token
    which_correct = (tokens == preds)[start:end]
    true = tokenizer.decode(tokens[start:end])
    pred = tokenizer.decode(preds[start:end])
    pred = color_sequence(pred, which_correct)

    data_logger.info(f"pred ({start}-{end}): " + " ".join(pred))
    data_logger.info(f"true ({start}-{end}): " + " ".join(true))
    return None


def log_metadata(args, model, loaders, state, checkpoint_savename) -> None:
    logger.info("Args:\n%s\n", pprint.pformat(vars(args)))
    logger.info("\n")
    if args.restore_checkpoint_from is not None:
        logger.info("Model config restored from checkpoint (overrides args):"
                    f"\n{pprint.pformat(model.config)}\n")

    logger.info(f"Tags: {args.tags}")
    for name, loader in loaders.items():
        logger.info(f"Number of {name} examples: {loader.shape['tokens'][0]:,}.")
        logger.info(f"Data shapes for {name}: "
                    f"weights: {loader.shape['weights']}, "
                    f"tokens: {loader.shape['tokens']}")
    logger.info(f"Max number of datapoints to load: {args.ndata:,}.")
    logger.info(f"Number of parameters in meta-model: "
                f"{count_params(state.params) / 1e6} Million")
    
    logger.info(f"Expected sequence length: {model.config.max_len}.")
    if args.save_checkpoint:
        logger.info(f"When training is completed, save final checkpoint "
                    f"to {args.checkpoint_dir}/{checkpoint_savename}.")
    print()
    return None

