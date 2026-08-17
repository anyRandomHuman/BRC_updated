import functools

import jax
import jax.numpy as jnp
import numpy as np

from jaxrl.agent.brc_learner import BRC
from jaxrl.agent.update import (
    _apply_model_gradients,
    build_actor_input,
    update_target_critic,
    update_temperature,
)
from jaxrl.utils import Batch, Params, PRNGKey


def _segment_counts(task_ids, num_tasks, dtype=jnp.float32):
    task_ids = jnp.asarray(task_ids, dtype=jnp.int32).reshape(-1)
    return jax.ops.segment_sum(
        jnp.ones(task_ids.shape, dtype=dtype),
        task_ids,
        num_segments=num_tasks,
    )


def _segment_mean(values, task_ids, num_tasks):
    """Mean values by task while preserving non-batch feature dimensions."""
    values = jnp.asarray(values)
    task_ids = jnp.asarray(task_ids, dtype=jnp.int32).reshape(-1)
    sums = jax.ops.segment_sum(values, task_ids, num_segments=num_tasks)
    counts = _segment_counts(task_ids, num_tasks, values.dtype)
    denominator_shape = (num_tasks,) + (1,) * (values.ndim - 1)
    return sums / jnp.maximum(counts, 1).reshape(denominator_shape)


def _segment_feature_mean(values, task_ids, num_tasks):
    """Return one scalar mean per task for a batch-first tensor."""
    task_values = _segment_mean(values, task_ids, num_tasks)
    if task_values.ndim == 1:
        return task_values
    return task_values.mean(axis=tuple(range(1, task_values.ndim)))


def _segment_feature_extreme(values, task_ids, num_tasks, use_max):
    """Return one scalar feature-and-batch extreme per task."""
    values = jnp.asarray(values)
    task_ids = jnp.asarray(task_ids, dtype=jnp.int32).reshape(-1)
    if use_max:
        task_values = jax.ops.segment_max(values, task_ids, num_segments=num_tasks)
        reducer = jnp.max
    else:
        task_values = -jax.ops.segment_max(-values, task_ids, num_segments=num_tasks)
        reducer = jnp.min
    if task_values.ndim == 1:
        return task_values
    return reducer(task_values, axis=tuple(range(1, task_values.ndim)))


def dynamic_critic_loss_fn(
    critic_params: Params,
    models,
    batch: Batch,
    key: PRNGKey,
    static_inputs,
):
    """Categorical critic loss, reduced to one value per transition first."""
    discount = static_inputs['discount']
    num_bins = static_inputs['num_bins']
    v_max = static_inputs['v_max']
    multitask = static_inputs['multitask']

    next_inputs = build_actor_input(
        models.critic, batch.next_observations, batch.task_ids, multitask
    )
    next_dist = models.actor(next_inputs)
    next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=key)
    next_q_logits = models.target_critic(
        batch.next_observations, next_actions, batch.task_ids
    )
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)

    v_min = -v_max
    bin_values = jnp.linspace(v_min, v_max, num_bins)
    delta_z = (v_max - v_min) / (num_bins - 1)
    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (
        bin_values[None] - models.temp() * next_log_probs[:, None]
    )
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_indices = (target_bin_values - v_min) / delta_z

    lower = jnp.floor(target_bin_indices)
    upper = jnp.ceil(target_bin_indices)
    lower_mask = jax.nn.one_hot(
        lower.reshape(-1), num_bins
    ).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(
        upper.reshape(-1), num_bins
    ).reshape((-1, num_bins, num_bins))
    lower_values = (
        next_q_probs
        * (upper + (lower == upper).astype(jnp.float32) - target_bin_indices)
    )[..., None]
    upper_values = (next_q_probs * (target_bin_indices - lower))[..., None]
    target_probs = jax.lax.stop_gradient(
        jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1)
    )

    q_logits = models.critic.apply(
        {'params': critic_params},
        batch.observations,
        batch.actions,
        batch.task_ids,
    )
    q_log_probs = jax.nn.log_softmax(q_logits, axis=-1)
    loss_per_critic_per_sample = -(
        target_probs[None] * q_log_probs
    ).sum(axis=-1)
    loss_per_sample = loss_per_critic_per_sample.mean(axis=0)
    critic_loss = loss_per_sample.mean()
    num_tasks = static_inputs['num_tasks']
    task_losses = _segment_mean(
        loss_per_sample, batch.task_ids, num_tasks
    )
    task_counts = _segment_counts(batch.task_ids, num_tasks, loss_per_sample.dtype)
    task_sample_fractions = task_counts / jnp.maximum(task_counts.sum(), 1)
    task_q_logits = _segment_mean(
        q_logits.mean(axis=0), batch.task_ids, num_tasks
    )

    q_value_target = (bin_values[None] * target_probs).sum(axis=-1)
    critic_entropy_per_sample = -(
        jax.nn.softmax(q_logits, axis=-1) * q_log_probs
    ).sum(axis=-1).mean(axis=0)
    batch_first_q_logits = jnp.moveaxis(q_logits, 1, 0)
    return critic_loss, {
        'critic_loss': critic_loss,
        'critic_loss_per_task': task_losses,
        'task_sample_counts': task_counts,
        'task_sample_fractions': task_sample_fractions,
        'q_mean': _segment_mean(q_value_target, batch.task_ids, num_tasks),
        'q_min': _segment_feature_extreme(
            q_value_target, batch.task_ids, num_tasks, use_max=False
        ),
        'q_max': _segment_feature_extreme(
            q_value_target, batch.task_ids, num_tasks, use_max=True
        ),
        'r': _segment_mean(batch.rewards, batch.task_ids, num_tasks),
        'critic_entropy': _segment_mean(
            critic_entropy_per_sample, batch.task_ids, num_tasks
        ),
        'q_logits': task_q_logits,
        'max_logit_before_softmax': _segment_feature_extreme(
            batch_first_q_logits, batch.task_ids, num_tasks, use_max=True
        ),
        'mean_logits_before_softmax': _segment_feature_mean(
            batch_first_q_logits, batch.task_ids, num_tasks
        ),
    }


def dynamic_actor_loss_fn(
    actor_params: Params,
    models,
    batch: Batch,
    key: PRNGKey,
    static_inputs,
):
    """Actor loss reduced with a single mean over sampled transitions."""
    inputs = build_actor_input(
        models.critic,
        batch.observations,
        batch.task_ids,
        static_inputs['multitask'],
    )
    dist = models.actor.apply({'params': actor_params}, inputs)
    actions, log_probs = dist.sample_and_log_prob(seed=key)
    q_logits = models.critic(batch.observations, actions, batch.task_ids)
    q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
    bin_values = jnp.linspace(
        -static_inputs['v_max'], static_inputs['v_max'], static_inputs['num_bins']
    )
    q_values = (bin_values[None] * q_probs).sum(axis=-1)

    loss_per_sample = log_probs * models.temp() - q_values
    actor_loss = loss_per_sample.mean()
    task_losses = _segment_mean(
        loss_per_sample, batch.task_ids, static_inputs['num_tasks']
    )
    return actor_loss, {
        'actor_loss': actor_loss,
        'actor_loss_per_task': task_losses,
        'actor_entropy': -log_probs.mean(),
        'actor_entropy_per_task': _segment_mean(
            -log_probs, batch.task_ids, static_inputs['num_tasks']
        ),
    }


def _simple_update(rng, models, batch, static_inputs):
    rng, critic_key, actor_key = jax.random.split(rng, 3)

    (critic_loss, critic_info), critic_grads = jax.value_and_grad(
        dynamic_critic_loss_fn, has_aux=True
    )(models.critic.params, models, batch, critic_key, static_inputs)
    del critic_loss
    new_critic, critic_grad_norm = _apply_model_gradients(
        models.critic, critic_grads
    )
    models = models.replace(critic=new_critic)

    (actor_loss, actor_info), actor_grads = jax.value_and_grad(
        dynamic_actor_loss_fn, has_aux=True
    )(models.actor.params, models, batch, actor_key, static_inputs)
    del actor_loss
    new_actor, actor_grad_norm = _apply_model_gradients(models.actor, actor_grads)
    models = models.replace(actor=new_actor)

    new_target_critic = update_target_critic(
        models.critic, models.target_critic, static_inputs['tau']
    )
    new_temp, temperature_info = update_temperature(
        models.temp, actor_info['actor_entropy'], static_inputs['target_entropy']
    )
    models = models.replace(target_critic=new_target_critic, temp=new_temp)

    info = {
        **critic_info,
        **actor_info,
        **temperature_info,
        'critic_gnorm': critic_grad_norm,
        'actor_gnorm': actor_grad_norm,
    }
    return rng, models, info


@functools.partial(jax.jit, static_argnames=('static_inputs',))
def _do_dynamic_updates(rng, models, batches, static_inputs, step):
    def one_step(update_index, state):
        current_step, current_rng, current_models, _ = state
        batch = jax.tree.map(lambda x: x[update_index], batches)
        new_rng, new_models, info = _simple_update(
            current_rng, current_models, batch, static_inputs
        )
        return current_step + 1, new_rng, new_models, info

    inputs = (step, rng, models, {})
    outs = one_step(0, inputs)
    return jax.lax.fori_loop(
        1,
        static_inputs['num_updates'],
        one_step,
        outs,
    )


class DynamicSamplingLearner(BRC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dynamic_cfg = getattr(self.cfg, 'dynamic_sampling', {})
        self.sampling_loss_ema = float(
            getattr(dynamic_cfg, 'loss_ema', 0.9)
        )
        self.sampling_temperature = float(
            getattr(dynamic_cfg, 'temperature', 1.0)
        )
        self.min_sampling_probability = float(
            getattr(dynamic_cfg, 'min_probability', 0.01)
        )
        if not 0 <= self.sampling_loss_ema < 1:
            raise ValueError("dynamic_sampling.loss_ema must be in [0, 1).")
        if self.sampling_temperature <= 0:
            raise ValueError("dynamic_sampling.temperature must be positive.")
        if not 0 <= self.min_sampling_probability <= 1 / self.num_tasks:
            raise ValueError(
                "dynamic_sampling.min_probability must be between 0 and 1 / num_tasks."
            )
        self._sampling_scores = np.ones(self.num_tasks, dtype=np.float64)
        self._sampling_proportions = np.ones(
            self.num_tasks, dtype=np.float64
        ) / self.num_tasks

        static_inputs = dict(self.static_inputs)
        static_inputs['num_tasks'] = self.num_tasks
        self.static_inputs = type(self.static_inputs)(static_inputs)

    def get_sampling_proportions(self):
        return self._sampling_proportions.copy()

    def _update_sampling_proportions(self, info):
        losses = np.asarray(info['critic_loss_per_task'], dtype=np.float64)
        counts = np.asarray(info['task_sample_counts'])
        valid = (counts > 0) & np.isfinite(losses)
        if not np.any(valid):
            return

        safe_losses = np.maximum(losses, 0.0)
        self._sampling_scores[valid] = (
            self.sampling_loss_ema * self._sampling_scores[valid]
            + (1 - self.sampling_loss_ema) * safe_losses[valid]
        )
        logits = np.log(np.maximum(self._sampling_scores, 1e-8))
        logits = logits / self.sampling_temperature
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        floor = self.min_sampling_probability
        self._sampling_proportions = (
            floor + (1 - floor * self.num_tasks) * probabilities
        )

    def update(self, batch: Batch, num_updates: int, env_step: int):
        if num_updates != self.static_inputs['num_updates']:
            raise ValueError(
                "num_updates must match the updates_per_step used to initialize the learner."
            )
        step, rng, self.models, info = _do_dynamic_updates(
            self.rng,
            self.models,
            batch,
            self.static_inputs,
            env_step,
        )
        self.step = step
        self.rng = rng
        self.actor = self.models.actor
        self.critic = self.models.critic
        self.target_critic = self.models.target_critic
        self.temp = self.models.temp

        self._update_sampling_proportions(info)
        info['task_sampling_proportions'] = jnp.asarray(
            self._sampling_proportions
        )
        return info
