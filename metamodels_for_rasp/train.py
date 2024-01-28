from collections import defaultdict

import numpy as np
import jax
from jax import random, jit, value_and_grad
import optax
import chex
import functools
from typing import Mapping, Any
from jax.typing import ArrayLike
import flax.linen as nn
import wandb

from metamodels_for_rasp.logger_config import setup_logger


logger = setup_logger(__name__)


# Optimizer and update function
@chex.dataclass
class TrainState:
    step: int
    rng: random.PRNGKey
    opt_state: optax.OptState
    params: dict


@chex.dataclass(frozen=True)  # needs to be immutable to be hashable (for static_argnums)
class Updater:
    """Holds training methods. All methods are jittable."""
    opt: optax.GradientTransformation
    model: nn.Module
    loss_fn: callable

    @functools.partial(jit, static_argnums=0)
    def get_metrics_and_loss(self, rng, params, data):
        """Compute acc and loss on a test set."""
        loss, aux = self.loss_fn(params, rng, data, is_training=False)
        out = {"loss": loss}
        if "metrics" in aux:
            out.update(aux["metrics"])
        return out, aux

    def init_train_state(self, rng: ArrayLike, inputs: ArrayLike) -> dict:
        out_rng, subkey = jax.random.split(rng)
        v = self.model.init(subkey, inputs, is_training=False)
        opt_state = self.opt.init(v["params"])
        if list(v.keys()) != ["params"]:
            raise ValueError("Expected model.init to return a dict with "
                f"a single key 'params'. Instead got {v.keys()}.")
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=v["params"],
        )
    
    @functools.partial(jit, static_argnums=0)
    def update(self, state: TrainState, data) -> (TrainState, dict):
        state.rng, subkey = jax.random.split(state.rng)
        (loss, aux), grads = value_and_grad(self.loss_fn, has_aux=True)(
                state.params, subkey, data)
        updates, state.opt_state = self.opt.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        metrics = {
                "train/loss": loss,
                "step": state.step,
                "grad_norm": optax.global_norm(grads),
                "weight_norm": optax.global_norm(state.params),
        }
        if "metrics" in aux:
            metrics.update({f"train/{k}": v for k, v in aux["metrics"].items()})
        metrics.update({
            f"opt/{k}": v for k, v in state.opt_state.hyperparams.items()
            })
        state.step += 1
        return state, metrics

    def compute_val_metrics(self, 
                            state: TrainState, 
                            data: dict,
                            name="val") -> (TrainState, dict):
        state.rng, subkey = random.split(state.rng)
        metrics, aux = self.get_metrics_and_loss(subkey, state.params, data)
        metrics = {f"{name}/{k}": v for k, v in metrics.items()}
        return state, metrics, aux


def print_metrics(metrics: Mapping[str, Any], prefix: str = ""):
    """Prints metrics to stdout. Assumption: metrics is a dict of scalars
    and always contains the keys "step" and "epoch".
    """
    for k, v in metrics.items():
        try:
            metrics[k] = np.round(v.item(), 7)
        except AttributeError:
            metrics[k] = np.round(v, 7)
        
    output = prefix
    output += f"Step: {metrics['step']}, Epoch: {metrics['epoch']}, "
    other_metrics = [k for k in metrics if k not in ["step", "epoch"]]
    output += ", ".join([f"{k}: {metrics[k]:.4f}" for k in other_metrics])
    logger.info(output)


@chex.dataclass
class Logger:
    """Helper class for logging metrics to wandb."""
    def __post_init__(self):
        self.metrics = defaultdict(list)
    
    def flush_mean(self, state, name="train", verbose=True, 
                   extra_metrics=None):
        """
        1) computes the mean of all metrics in self.metrics[name] 
        and extra_metrics
        2) logs the mean to wandb
        """
        metrics = self.metrics[name]
        if len(metrics) == 0:
            raise ValueError(f"No metrics currently logged in metrics[{name}]")

        # reduce
        means = {}
        for k in metrics[0].keys():
            means[k] = np.mean([d[k] for d in metrics])
        means["step"] = int(state.step)

        # update
        if extra_metrics is not None:
            means.update(extra_metrics)
        
        # log
        wandb.log(means, step=state.step)
        if verbose:
            print_metrics(means)
        
        # reset
        del self.metrics[name]

    def write(self, state, metrics, name="train"):
        """Add metrics to self.metrics[name]"""
        metrics["step"] = int(state.step)
        self.metrics[name].append(metrics)
    
    def get_metrics(self, name="train", metric="train/loss"):
        """returns a tuple (steps, metric)"""
        metrics_list = self.metrics[name]
        return zip(*[(d['step'], d[metric]) for d in metrics_list if metric in d])

