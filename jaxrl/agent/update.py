import functools
import jax.numpy as jnp
import jax
import optax
from jax import tree_map

from jaxrl.utils import Batch, Model, Params, PRNGKey, tree_norm, flatten_grads, compute_normalized_gram

@functools.partial(jax.jit, static_argnames=('multitask'))
def build_actor_input(critic: Model, observations: jnp.ndarray, task_ids: jnp.ndarray, multitask: bool):
    inputs = observations
    if multitask:
        task_embeddings = critic(None, None, task_ids, True)
        inputs = jnp.concatenate((inputs, task_embeddings), axis=-1)
    return inputs

def normalize_task_weights(weights: jnp.ndarray, eps: float = 1e-8):
    weights = jnp.clip(weights, a_min=eps)
    return weights / jnp.maximum(weights.sum(), eps)

def _apply_model_gradients(model: Model, grads):
    grad_norm = tree_norm(grads)
    updates, new_opt_state = model.tx.update(grads, model.opt_state, model.params)
    new_params = optax.apply_updates(model.params, updates)
    new_model = model.replace(
        step=model.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    return new_model, grad_norm

def _optimize_task_weights(loss_fn, num_tasks, static_inputs):
    w_init = jnp.ones(num_tasks) / num_tasks
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(learning_rate=static_inputs['lr'], momentum=static_inputs['momentum'])
    )
    opt_state = optimizer.init(w_init)

    def step_fn(state, _):
        w, current_opt_state, w_best, obj_best = state
        obj, grad_w = jax.value_and_grad(loss_fn)(w)

        is_better = obj < obj_best
        w_best = jnp.where(is_better, w, w_best)
        obj_best = jnp.where(is_better, obj, obj_best)

        updates, new_opt_state = optimizer.update(grad_w, current_opt_state, w)
        w_new = optax.apply_updates(w, updates)
        w_new = jnp.clip(w_new, min=1e-8)
        return (w_new, new_opt_state, w_best, obj_best), None

    init_state = (w_init, opt_state, w_init, jnp.inf)
    final_state, _ = jax.lax.scan(step_fn, init_state, None, length=static_inputs['niter'])
    _, _, w_best, _ = final_state
    return jax.lax.stop_gradient(w_best)

def default_weight_interpolation_fn(last_weights: jnp.ndarray, raw_weights: jnp.ndarray):
    weights = normalize_task_weights(raw_weights)
    delta = jnp.linalg.norm(weights - last_weights)
    return weights, raw_weights, delta


def _interpolate_and_aggregate(task_grads, last_weights, raw_weights, static_inputs):
    weight_interpolation_fn = static_inputs.get('weight_interpolation_fn', default_weight_interpolation_fn)
    weights, raw_w, delta = weight_interpolation_fn(last_weights, raw_weights)

    def aggregate_leaf(leaf):
        return jnp.tensordot(weights, leaf, axes=(0, 0))

    combined_grads = tree_map(aggregate_leaf, task_grads)
    return combined_grads, weights, raw_w, delta


def _spectral_metrics(gram_matrix, n_tasks):
    eigen_values = jnp.linalg.eigvalsh(gram_matrix)
    return {
        'conflict_ratio': (gram_matrix < 0).sum() / (n_tasks ** 2),
        'condition_number': eigen_values[-1] / eigen_values[0],
        'min_eigen': eigen_values[0],
        'max_eigen': eigen_values[-1],
        'mean_eigen': eigen_values.mean(),
    }

def mean_grad_process(task_grads, previous_weights: jnp.ndarray, key: PRNGKey, static_inputs):
    del key, static_inputs

    num_tasks = previous_weights.shape[0]
    raw_weights = jnp.ones((num_tasks,), dtype=previous_weights.dtype) / jnp.maximum(num_tasks, 1)
    task_weights = normalize_task_weights(raw_weights)
    weight_delta = jnp.linalg.norm(task_weights - previous_weights)

    merged_grads = jax.tree.map(
        lambda leaf: jnp.tensordot(task_weights, leaf, axes=(0, 0)),
        task_grads
    )

    task_metrics = {
        'task_weights': task_weights,
        'raw_weights': raw_weights,
    }
    process_metrics = {
        'weight_delta': weight_delta,
    }
    return merged_grads, task_metrics, process_metrics

def fairgrad(task_grads, previous_weights, key, static_inputs):
    """
    JAX implementation of FairGrad.

    Args:
        task_grads: A PyTree where every leaf has shape (n_tasks, ...).
                    This represents the gradients per task.
        alpha: Fairness hyperparameter (float).
        niter: Number of optimization steps to find weights w.
        lr: Learning rate for the weight optimization.
        momentum: Momentum for the SGD optimizer.

    Returns:
        A PyTree with the same structure as task_grads (minus the leading n_tasks dim),
        containing the aggregated gradient direction.
    """

    del key
    alpha = static_inputs['alpha']
    G = flatten_grads(task_grads)
    n_tasks = G.shape[0]
    GG, _ = compute_normalized_gram(G)

    def loss_fn(w):
        w_safe = jnp.clip(w, a_min=1e-8)
        residual = jnp.dot(GG, w_safe) - (w_safe ** (-1.0 / alpha))
        return jnp.sum(residual ** 2)

    w_best = _optimize_task_weights(loss_fn, n_tasks, static_inputs)
    combined_grads, weights, raw_w, delta = _interpolate_and_aggregate(task_grads, previous_weights, w_best, static_inputs)

    task_metrics = {
        'task_weights': weights,
        'raw_weights': raw_w,
    }
    metrics = {
        'weight_delta': delta,
        **_spectral_metrics(GG, n_tasks),
    }

    return combined_grads, task_metrics, metrics

_GRAD_PROCESSORS = {
    'mean': mean_grad_process,
    'fairgrad': fairgrad,
}

def resolve_grad_process_fn(grad_process: str):
    if grad_process not in _GRAD_PROCESSORS:
        supported = ', '.join(sorted(_GRAD_PROCESSORS))
        raise ValueError(f"Unsupported grad_process: {grad_process}. Supported: {supported}")
    return _GRAD_PROCESSORS[grad_process]

def process_task_gradients(task_grads, previous_weights: jnp.ndarray, key: PRNGKey, static_inputs):
    grad_process_fn = static_inputs.get('grad_process_fn')
    if grad_process_fn is None:
        grad_process_fn = resolve_grad_process_fn(static_inputs.get('grad_process', 'mean'))
    return grad_process_fn(task_grads, previous_weights, key, static_inputs)
# def update_actor(key: PRNGKey, models, batch: Batch, static_inputs):
#     actor = models.actor
#     critic = models.critic
#     temp = models.temp
#
#     num_bins = static_inputs['num_bins']
#     v_max = static_inputs['v_max']
#     multitask = static_inputs['multitask']
#
#     inputs = build_actor_input(critic, batch.observations, batch.task_ids, multitask)
#     def actor_loss_fn(actor_params: Params):
#         dist = actor.apply({'params': actor_params}, inputs)
#         actions, log_probs = dist.sample_and_log_prob(seed=key)
#         q_logits = critic(batch.observations, actions, batch.task_ids)
#         q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
#         bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[None]
#         q_values = (bin_values * q_probs).sum(-1)
#         actor_loss = (log_probs * temp().mean() - q_values).mean()
#         return actor_loss, {
#             'actor_loss': actor_loss,
#             'actor_entropy': -log_probs.mean(),
#             'actor_pnorm': tree_norm(actor_params),
#         }
#     new_actor, info = actor.apply_gradient(actor_loss_fn)
#     info['actor_gnorm'] = info.pop('grad_norm')
#     return new_actor, info

# def update_critic(key: PRNGKey, models, batch: Batch, static_inputs):
#     actor = models.actor
#     critic = models.critic
#     target_critic = models.target_critic
#     temp = models.temp
#
#     discount = static_inputs['discount']
#     num_bins = static_inputs['num_bins']
#     v_max = static_inputs['v_max']
#     multitask = static_inputs['multitask']
#
#     inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
#     dist = actor(inputs)
#     next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
#     next_q_logits = target_critic(batch.next_observations, next_actions, batch.task_ids)
#     next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
#     v_min = -v_max
#     bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]
#
#     delta_z = ((v_max - v_min) / (num_bins - 1))
#     target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (bin_values - temp() * next_log_probs[:, None])
#     target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
#     target_bin_values = (target_bin_values - v_min) / delta_z
#
#     lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
#     lower_mask = jax.nn.one_hot(lower.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
#     upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
#
#     lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]
#     upper_values = (next_q_probs * (target_bin_values - lower))[..., None]
#
#     target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
#     q_value_target = (bin_values * target_probs).sum(-1)
#     def critic_loss_fn(critic_params: Params):
#         q_logits = critic.apply({"params": critic_params}, batch.observations, batch.actions, batch.task_ids)
#         q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
#         critic_loss = -(target_probs[None] * q_logprobs).sum(-1).mean(-1).sum(-1)
#         critic_entropy = - (jax.nn.softmax(q_logits) * q_logprobs).sum(axis=-1).mean()
#         return critic_loss, {
#             "critic_loss": critic_loss,
#             "q_mean": q_value_target.mean(),
#             "q_min": q_value_target.min(),
#             "q_max": q_value_target.max(),
#             "r": batch.rewards.mean(),
#             "critic_pnorm": tree_norm(critic_params),
#             'q_logits': q_logits,
#             'critic_entropy': critic_entropy,
#         }
#     new_critic, info = critic.apply_gradient(critic_loss_fn)
#     info["critic_gnorm"] = info.pop("grad_norm")
#     return new_critic, info

def update_target_critic(critic: Model, target_critic: Model, tau: float):
    new_target_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)
    return target_critic.replace(params=new_target_params)

def update_temperature(temp: Model, entropy: float, target_entropy: float):
    def temperature_loss_fn(temp_params):
        temperature = temp.apply({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')
    return new_temp, info

def critic_loss_fn(critic_params: Params, models, batch, key, static_inputs):
    actor = models.actor
    critic = models.critic
    target_critic = models.target_critic
    temp = models.temp

    discount = static_inputs['discount']
    num_bins = static_inputs['num_bins']
    v_max = static_inputs['v_max']
    multitask = static_inputs['multitask']

    inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
    dist = actor(inputs)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q_logits = target_critic(batch.next_observations, next_actions, batch.task_ids)
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    v_min = -v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]

    delta_z = ((v_max - v_min) / (num_bins - 1))
    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (
                bin_values - temp() * next_log_probs[:, None])
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / delta_z

    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))

    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]

    target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
    q_value_target = (bin_values * target_probs).sum(-1)
    q_logits = critic.apply({"params": critic_params}, batch.observations, batch.actions, batch.task_ids)
    q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)

    critic_loss = -(target_probs[None] * q_logprobs).sum(-1).sum(axis=0).mean()
    critic_entropy = - (jax.nn.softmax(q_logits, axis=-1) * q_logprobs).sum(axis=-1).mean()

    return critic_loss, {
        "critic_loss": critic_loss,
        "q_mean": q_value_target.mean(),
        "q_min": q_value_target.min(),
        "q_max": q_value_target.max(),
        "r": batch.rewards.mean(),
        'critic_entropy': critic_entropy,
        'q_logits': q_logits.mean(axis=(0,1)),
    },

def update_critic(key: PRNGKey, models, batch: Batch, static_inputs):
    if static_inputs.get('do_separate_critic_grad', False):
        grad_key, process_key = jax.random.split(key, 2)
        grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        vmap_grad_fn = jax.vmap(grad_fn, in_axes=(None, None, 0, None, None))
        (_, task_metrics), task_grads = vmap_grad_fn(
            models.critic.params, models, batch, grad_key, static_inputs
        )

        merged_grads, task_process_metrics, process_metrics = process_task_gradients(
            task_grads, models.critic_grad_weights, process_key, static_inputs
        )
        new_critic, grad_norm = _apply_model_gradients(models.critic, merged_grads)
        models = models.replace(
            critic=new_critic,
            critic_grad_weights=task_process_metrics['task_weights'],
        )

        info = {
            **task_metrics,
            'critic_task_weights': task_process_metrics['task_weights'],
            'critic_task_weights_raw': task_process_metrics['raw_weights'],
            'critic_grad_norms': jax.vmap(tree_norm)(task_grads),
            **{f'critic_{k}': v for k, v in process_metrics.items()},
        }
        info['critic_gnorm'] = grad_norm
        return models, info

    loss_process = static_inputs['loss_process']

    def vmap_loss_fn(critic_params: Params):
        fn = jax.vmap(critic_loss_fn, in_axes=(None, None, 0, None, None))
        task_loss, task_metrics = fn(critic_params, models, batch, key, static_inputs)
        if static_inputs['warmup_done'] and static_inputs['balance_critic']:
            if 'famo' in loss_process:
                weights = jax.nn.softmax(models.cw_state.params, -1)
                co = jax.lax.stop_gradient((weights / (task_loss + 1e-8)).sum())
                weighted_loss = (weights * jnp.log(task_loss + 1e-8) / co)
                critic_loss = weighted_loss.mean()

                task_metrics = {**task_metrics, 'critic_task_weights': weights}
            elif loss_process == 'inverse_scale':
                weights = jnp.mean(task_loss) / task_loss
                weighted_loss = jax.lax.stop_gradient(weights) * task_loss
                critic_loss = weighted_loss.mean()
            elif loss_process == 'mean':
                critic_loss = jnp.mean(task_loss)
        else:
            critic_loss = jnp.mean(task_loss)
        metrics = {
            "critic_pnorm": tree_norm(critic_params),
        }
        return critic_loss, task_metrics

    new_critic, info = models.critic.apply_gradient(vmap_loss_fn)

    models = models.replace(critic=new_critic)

    if static_inputs['warmup_done'] and static_inputs['balance_critic']:
        if 'famo' in loss_process:
            cw_state = models.cw_state

            if loss_process == 'famo_total':
                task_loss = models.critic_loss
            else:
                task_loss = info['critic_loss']

            _, new_info = vmap_loss_fn(new_critic.params)
            updated_task_loss = info['critic_loss']
            delta = jax.lax.stop_gradient(jnp.log(task_loss + 1e-8) - jnp.log(updated_task_loss + 1e-8))

            def softmax_fn(params):
                return jax.nn.softmax(params, axis=-1)

            softmax_out, vjp_fun = jax.vjp(softmax_fn, cw_state.params)
            d = vjp_fun(delta)[0]  # the return is a tuple
            cw_state = cw_state.apply_gradients(grads=d)

            models = models.replace(cw_state=cw_state)

    info["critic_gnorm"] = info.pop("grad_norm")

    return models, info

def actor_loss_fn(actor_params: Params, models, batch: Batch, key: PRNGKey, static_inputs):
    num_bins = static_inputs['num_bins']
    v_max = static_inputs['v_max']
    multitask = static_inputs['multitask']

    actor = models.actor
    critic = models.critic
    temp = models.temp

    inputs = build_actor_input(models.critic, batch.observations, batch.task_ids, multitask)

    dist = actor.apply({'params': actor_params}, inputs)
    actions, log_probs = dist.sample_and_log_prob(seed=key)
    q_logits = critic(batch.observations, actions, batch.task_ids)
    q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
    bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[None]
    q_values = (bin_values * q_probs).sum(-1)
    actor_loss = (log_probs * temp().mean() - q_values).mean()
    return actor_loss, {
        'actor_loss': actor_loss,
        'actor_entropy': -log_probs.mean(),
    }



def update_actor(key: PRNGKey, models, batch: Batch, static_inputs):
    if static_inputs.get('do_separate_actor_grad', False):
        grad_key, process_key = jax.random.split(key, 2)
        grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
        vmap_grad_fn = jax.vmap(grad_fn, in_axes=(None, None, 0, None, None))
        (_, task_metrics), task_grads = vmap_grad_fn(
            models.actor.params, models, batch, grad_key, static_inputs
        )

        merged_grads, task_process_metrics, process_metrics = process_task_gradients(
            task_grads, models.actor_grad_weights, process_key, static_inputs
        )
        new_actor, grad_norm = _apply_model_gradients(models.actor, merged_grads)
        models = models.replace(
            actor=new_actor,
            actor_grad_weights=task_process_metrics['task_weights'],
        )

        info = {
            **task_metrics,
            'actor_task_weights': task_process_metrics['task_weights'],
            'actor_task_weights_raw': task_process_metrics['raw_weights'],
            'actor_grad_norms': jax.vmap(tree_norm)(task_grads),
            **{f'actor_{k}': v for k, v in process_metrics.items()},
        }
        info['actor_gnorm'] = grad_norm
        return models, info

    loss_process = static_inputs['loss_process']

    # new_args = (args[0], args[1], args[2], args[3], new_batch, args[5], args[6], args[7])
    def vmap_loss_fn(params):
        vmap_loss = jax.vmap(actor_loss_fn, in_axes=(None, None, 0, None, None))
        task_loss, task_metrics = vmap_loss(params, models, batch, key, static_inputs)

        metrics = {}
        if static_inputs['warmup_done']:
            if 'famo' in loss_process:
                weights = jax.nn.softmax(models.aw_state.params, -1)
                co = jax.lax.stop_gradient((weights / (task_loss + 1e-8)).sum())
                weighted_loss = (weights * jnp.log(task_loss + 1e-8) / co)
                actor_loss = weighted_loss.sum()

                task_metrics = {**task_metrics, 'actor_task_weights': weights}
            elif loss_process == 'mean':
                actor_loss = jnp.mean(task_loss)
            elif loss_process == 'inverse_scale':
                weights = jnp.mean(task_loss) / task_loss
                weighted_loss = jax.lax.stop_gradient(weights) * task_loss
                actor_loss = weighted_loss.mean()
            else:
                raise NotImplementedError
        else:
            actor_loss = jnp.mean(task_loss)

        return actor_loss, task_metrics

    new_actor, info = models.actor.apply_gradient(vmap_loss_fn)

    models = models.replace(actor=new_actor)

    if static_inputs['warmup_done']:
        if 'famo' in loss_process:
            if loss_process == 'famo_total':
                task_loss = models.actor_loss
            else:
                task_loss = info['actor_loss']
            aw_state = models.aw_state

            _, new_info = vmap_loss_fn(new_actor.params)
            updated_task_loss = info['actor_loss']
            delta = jax.lax.stop_gradient(jnp.log(jnp.abs(updated_task_loss) + 1e-8) - jnp.log(jnp.abs(task_loss) + 1e-8))

            def softmax_fn(params):
                return jax.nn.softmax(params, axis=-1)  # axis=-1对应原代码的dim=-1

            softmax_out, vjp_fun = jax.vjp(softmax_fn, aw_state.params)
            d = vjp_fun(delta)[0]  # the return is a tuple
            aw_state = aw_state.apply_gradients(grads=d)

            models = models.replace(aw_state=aw_state)

    info['actor_gnorm'] = info.pop('grad_norm')

    return models, info

'''
from jaxrl.utils import Batch

key = agent.rng
actor = agent.actor
target_critic = agent.target_critic
critic = agent.critic
temp = agent.temp
batch = Batch(
    observations=batches.observations[0],
    actions=batches.actions[0],
    rewards=batches.rewards[0],
    masks=batches.masks[0],
    next_observations=batches.next_observations[0],
    task_ids=batches.task_ids[0])
discount = agent.discount
num_bins = agent.num_bins
v_max = agent.v_max
multitask = agent.multitask
'''
