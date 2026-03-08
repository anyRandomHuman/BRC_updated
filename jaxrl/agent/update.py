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

def update_actor(key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch, num_bins: int, v_max: float, multitask: bool):
    inputs = build_actor_input(critic, batch.observations, batch.task_ids, multitask)
    def actor_loss_fn(actor_params: Params):
        dist = actor.apply({'params': actor_params}, inputs)        
        actions, log_probs = dist.sample_and_log_prob(seed=key)
        q_logits = critic(batch.observations, actions, batch.task_ids)        
        q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
        bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[None]
        q_values = (bin_values * q_probs).sum(-1)    
        actor_loss = (log_probs * temp().mean() - q_values).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'actor_pnorm': tree_norm(actor_params),
        }
    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['actor_gnorm'] = info.pop('grad_norm')
    return new_actor, info

def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, num_bins: int, v_max: float, multitask: bool):
    inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
    dist = actor(inputs)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q_logits = target_critic(batch.next_observations, next_actions, batch.task_ids)
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    v_min = -v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]
    
    delta_z = ((v_max - v_min) / (num_bins - 1))
    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (bin_values - temp() * next_log_probs[:, None])
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / delta_z
    
    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    
    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]        
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]
    
    target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
    q_value_target = (bin_values * target_probs).sum(-1)
    def critic_loss_fn(critic_params: Params):
        q_logits = critic.apply({"params": critic_params}, batch.observations, batch.actions, batch.task_ids)
        q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
        critic_loss = -(target_probs[None] * q_logprobs).sum(-1).mean(-1).sum(-1)
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q_value_target.mean(),
            "q_min": q_value_target.min(),
            "q_max": q_value_target.max(),
            "r": batch.rewards.mean(),
            "critic_pnorm": tree_norm(critic_params),
            'q_logits': q_logits,
        }
    new_critic, info = critic.apply_gradient(critic_loss_fn)
    info["critic_gnorm"] = info.pop("grad_norm")
    return new_critic, info

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


def update_critic_famo(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                  temp: Model, batch: Batch, discount: float, num_bins: int, v_max: float, multitask: bool, cw_state, famo):
    n_tasks = batch.rewards.shape[0]
    per_task_batch = batch.rewards.shape[1]

    obs = batch.observations.reshape((n_tasks * per_task_batch, -1))
    actions = batch.actions.reshape((n_tasks * per_task_batch, -1))
    rewards = batch.rewards.reshape((n_tasks * per_task_batch))
    next_obs = batch.next_observations.reshape((n_tasks * per_task_batch, -1))
    task_ids = batch.task_ids.reshape((n_tasks * per_task_batch))
    masks = batch.masks.reshape((n_tasks * per_task_batch))

    inputs = build_actor_input(critic, next_obs, task_ids, multitask)
    dist = actor(inputs)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_q_logits = target_critic(next_obs, next_actions, task_ids)
    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    v_min = -v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]

    delta_z = ((v_max - v_min) / (num_bins - 1))
    target_bin_values = rewards[:, None] + discount * masks[:, None] * (
                bin_values - temp() * next_log_probs[:, None])
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / delta_z

    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape((-1)), num_bins).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))

    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]

    target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
    q_value_target = (bin_values * target_probs).sum(-1)

    def critic_loss_fn(critic_params: Params):
        q_logits = critic.apply({"params": critic_params}, obs, actions, task_ids)
        q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
        task_loss = -(target_probs[None] * q_logprobs).sum(-1).sum(axis=0).reshape(n_tasks, per_task_batch).mean(axis=-1)
        # critic_loss = task_loss.mean()
        if famo == 1:
            weights = jax.nn.softmax(cw_state.params, -1)
            co = jax.lax.stop_gradient((weights / (task_loss + 1e-8)).sum())
            weighted_loss = (weights * jnp.log(task_loss + 1e-8) / co)
            critic_loss = weighted_loss.sum()
        elif famo == 2:
            weights = jnp.mean(task_loss) / task_loss
            weighted_loss = jax.lax.stop_gradient(weights) * task_loss
            critic_loss = weighted_loss.sum()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q_value_target.mean(),
            "q_min": q_value_target.min(),
            "q_max": q_value_target.max(),
            "r": rewards.mean(),
            "critic_pnorm": tree_norm(critic_params),
            'task_loss': task_loss,
        },

    # def vmap_loss_fn(critic_params: Params):
    #     fn = jax.vmap(critic_loss_fn, in_axes=(None, 0, 0, 0, 0))
    #     task_loss = fn(critic_params, batch.observations, batch.actions, batch.task_ids, batch.rewards)
    #     weights = jax.nn.softmax(cw_state.params, -1)
    #     co = jax.lax.stop_gradient((weights / (task_loss + 1e-8)).sum())
    #     weighted_loss = (weights * jnp.log(task_loss + 1e-8) / co)
    #     critic_loss = weighted_loss.sum()
    #     return critic_loss, ({}, task_loss)

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    if famo == 1:
        task_loss = info['task_loss']
        info["critic_gnorm"] = info.pop("grad_norm")

        _, new_info = critic_loss_fn(new_critic.params)
        updated_task_loss = info['task_loss']
        delta = jax.lax.stop_gradient(jnp.log(task_loss + 1e-8) - jnp.log(updated_task_loss + 1e-8))

        def softmax_fn(params):
            return jax.nn.softmax(params, axis=-1)  # axis=-1对应原代码的dim=-1

        softmax_out, vjp_fun = jax.vjp(softmax_fn, cw_state.params)
        d = vjp_fun(delta)[0]  # the return is a tuple
        cw_state = cw_state.apply_gradients(grads=d)
    if famo == 2:
        pass

    return new_critic, info, cw_state

def update_actor_famo(*args, **kwargs):
    batch = args[4]
    new_shape = (batch.observations.shape[0] * batch.observations.shape[1], -1)

    new_batch = Batch(
        observations=batch.observations.reshape(new_shape),
        actions=batch.actions.reshape(new_shape),
        rewards=batch.rewards.reshape(batch.observations.shape[0] * batch.observations.shape[1]),
        masks=batch.masks.reshape(batch.observations.shape[0] * batch.observations.shape[1]),
        next_observations=batch.next_observations.reshape(new_shape),
        task_ids=batch.task_ids.reshape(batch.observations.shape[0] * batch.observations.shape[1]),
    )
    new_args = (args[0], args[1], args[2], args[3], new_batch, args[5], args[6], args[7])
    return update_actor(*new_args, **kwargs)
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
