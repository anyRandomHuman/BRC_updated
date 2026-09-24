import os
import collections
from typing import Any, Optional, Sequence

import flax
from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp
import optax

Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'task_ids'])

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

def flatten_grads(grads):
    def flatten_single_task(grad_tree):
        return ravel_pytree(grad_tree)[0]
    return jax.vmap(flatten_single_task)(grads)

def compute_normalized_gram(G):
    norms = jnp.linalg.norm(G, axis=-1)
    max_norm = jnp.max(norms)
    normed_G = G / max_norm
    GG = jnp.matmul(normed_G, normed_G.T)
    return GG, max_norm

def task_weight_bounds(num_tasks, max_ratio=2.0):
    if max_ratio is None or max_ratio <= 0.0:
        return 0., 1.0
    lower = 1.0 / (1.0 + (num_tasks - 1) * max_ratio)
    upper = max_ratio / (max_ratio + num_tasks - 1)
    return lower, upper

def bound_temp(logits, lower, upper, k = None):
    """

    Args:
        logits:
        k: num of logits to normalize
        lower:
        upper:

    Returns:

    """

    if k is None:
        k = logits.shape[-1]
    if lower <= 0.:
        lower = 1e-8
    if upper >= 1.0:
        upper = 1 - 1e-8
    delta_z = jnp.max(logits, axis=0) - jnp.min(logits, axis=0)

    # 2. Calculate theoretical minimum temperatures for both bounds
    denom_L = jnp.log((1 - lower) / (lower * (k - 1)))
    denom_U = jnp.log((upper * (k - 1)) / (1 - upper))

    tau_L = delta_z / denom_L
    tau_U = delta_z / denom_U

    tau = jnp.maximum(1.0, jnp.maximum(tau_L, tau_U))
    return tau

@flax.struct.dataclass
class SaveState:
    params: Params
    opt_state: Optional[optax.OptState] = None


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: flax.linen.Module = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls,
               model_def: flax.linen.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None):
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        return self.apply_fn.apply({'params': self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.apply_fn.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        grad_norm = tree_norm(grads)
        info['grad_norm'] = grad_norm

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info
    
    def get_gradient(self, loss_fn):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        return grads

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(SaveState(params=self.params, opt_state=self.opt_state)))

    def load(self, load_path: str):
        with open(load_path, 'rb') as f:
            contents = f.read()
            saved_state = flax.serialization.from_bytes(
                SaveState(params=self.params, opt_state=self.opt_state), contents
            )
        return self.replace(params=saved_state.params, opt_state=saved_state.opt_state)