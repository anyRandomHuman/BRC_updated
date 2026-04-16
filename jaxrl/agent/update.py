import functools
import jax.numpy as jnp
import jax

from jaxrl.utils import Batch, Model, Params, PRNGKey, tree_norm

@functools.partial(jax.jit, static_argnames=('multitask'))
def build_actor_input(critic: Model, observations: jnp.ndarray, task_ids: jnp.ndarray, multitask: bool):
    inputs = observations
    if multitask:
        task_embeddings = critic(None, None, task_ids, True)
        inputs = jnp.concatenate((inputs, task_embeddings), axis=-1)
    return inputs

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
