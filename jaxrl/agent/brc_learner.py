import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from jaxrl.agent.update import (
    build_actor_input,
    default_weight_interpolation_fn,
    resolve_grad_process_fn,
    update_actor,
    update_critic,
    update_target_critic,
    update_temperature,
)

from jaxrl.networks import NormalTanhPolicy, Critic, Temperature
from jaxrl.utils import Model, PRNGKey, Batch

@struct.dataclass
class Models:
    critic: Model
    actor: Model
    temp: Model
    target_critic: Model
    cw_state: TrainState
    aw_state: TrainState
    critic_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    actor_grad_weights: jnp.ndarray
    critic_grad_weights: jnp.ndarray

@functools.partial(jax.jit, static_argnames=('discount', 'target_entropy', 'num_bins', 'v_max', 'multitask'),)
@functools.partial(jax.vmap, in_axes=(None, None, None, None, None, 0, None, None, None, None, None))
def _get_infos(
    rng: PRNGKey, 
    actor: Model, 
    critic: Model, 
    target_critic: Model, 
    temp: Model, 
    batch: Batch, 
    discount: float, 
    target_entropy: float, 
    num_bins: int, 
    v_max: float,
    multitask: bool
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    _, critic_info = update_critic(critic_key, actor, critic, target_critic, temp, batch, discount, num_bins, v_max, multitask)
    _, actor_info = update_actor(actor_key, actor, critic, temp, batch, num_bins, v_max, multitask) 
    _, alpha_info = update_temperature(temp, actor_info['actor_entropy'], target_entropy)
    return {
        **critic_info,
        **actor_info,
        **alpha_info,
    }

@jax.jit
def _get_temperature(temp):
    temp_val = temp()
    return temp_val
    
@jax.jit
def _sample_actions(
    rng: PRNGKey,
    actor: Model,
    inputs: np.ndarray,
    temperature: float = 1.0,
):
    dist = actor(inputs, temperature)
    rng, key = jax.random.split(rng)
    actions = dist.sample(seed=key)
    return rng, actions

def _update(
    rng: PRNGKey, 
    models,
    batch: Batch, 
    static_inputs,
):
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    models, critic_info = update_critic(critic_key, models, batch, static_inputs)
    models, actor_info = update_actor(actor_key, models, batch, static_inputs)
    new_target_critic = update_target_critic(models.critic, models.target_critic, static_inputs['tau'])
    new_temp, alpha_info = update_temperature(models.temp, actor_info['actor_entropy'], static_inputs['target_entropy'])

    models = models.replace(target_critic=new_target_critic, temp=new_temp)

    returns = (rng, models, {
        **critic_info,
        **actor_info,
        **alpha_info,
    })
    return returns

@functools.partial(jax.jit, static_argnames=('static_inputs'))
def _do_multiple_updates(
    rng: PRNGKey,
    models,
    batches: Batch,
    static_inputs,
    step,
):
    def one_step(i, state):
        step, rng, models, info = state
        step = step + 1
        returns = _update(
            rng,
            models,
            jax.tree.map(lambda x: jnp.take(x, i, axis=0), batches),
            static_inputs
        )

        return step, *returns

    inputs = (step, rng, models, {})
    outs = one_step(0, inputs)
    return jax.lax.fori_loop(1, static_inputs['num_updates'], one_step, outs)

class BRC(object):
    def __init__(
        self,
        seed: int,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        num_tasks: int,
        embedding_size: int = 32,
        ensemble_size: int = 2,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        init_temperature: float = 0.1,
        updates_per_step: int = 10,
        width_critic: int = 512,
        width_actor: int = 256,
        num_bins: int = 101,
        v_max: float = 10.0,
        w_lr =  0.025,
        w_d = 0.99,
        loss_process = 'mean',
        warmup_epochs: int = 10000,
        use_separate_actor_grad: bool = True,
        use_separate_critic_grad: bool = True,
        separate_grad_every: int = 1,
        grad_process: str = 'mean',
        grad_process_fn: Optional[Callable] = None,
        grad_process_alpha= 1.0,
        grad_lr = 0.1,
        grad_momentum = 0.5,
        niter = 20,
        last_init_str: str = 'orthogonal',
        depth_actor=2,
        cfg=None,
    ) -> None:
        
        action_dim = actions.shape[-1]
        self.action_dim = float(action_dim)
        self.seed = seed
        self.target_entropy = -self.action_dim / 2 if target_entropy is None else target_entropy
        self.tau = tau
        self.discount = discount
        self.num_bins = num_bins
        self.v_max = v_max
        
        self.num_tasks = num_tasks
        self.embedding_size = embedding_size
        self.task_ids = jnp.arange(num_tasks, dtype=jnp.int32)
        
        task_embedding_init = jnp.zeros((1, embedding_size))
        task_ids_init = self.task_ids[:1]
        self.multitask = True if num_tasks > 1 else False
        
        actor_init = jnp.concatenate((observations, task_embedding_init), axis=-1) if self.multitask else observations
        
        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
            actor_def = NormalTanhPolicy(action_dim=action_dim, hidden_dims=width_actor, depth=depth_actor)
            critic_def = Critic(
                num_tasks=num_tasks, embedding_size=embedding_size, ensemble_size=ensemble_size,
                hidden_dims=width_critic, depth=2, output_nodes=num_bins, multitask=self.multitask,
                last_init_str=last_init_str
            )
            actor = Model.create(actor_def, inputs=[actor_key, actor_init], tx=optax.adamw(learning_rate=actor_lr))
            critic = Model.create(critic_def, inputs=[critic_key, observations, actions, task_ids_init], tx=optax.adamw(learning_rate=critic_lr))
            target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions, task_ids_init])
            temp = Model.create(Temperature(init_temperature), inputs=[temp_key], tx=optax.adam(learning_rate=temp_lr, b1=0.5))
            return actor, critic, target_critic, temp, rng

        self.init_models = jax.jit(_init_models)
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seed)
        self.step = 1

        from flax.training.train_state import TrainState
        cw_state = TrainState.create(apply_fn=None, params=jnp.zeros(num_tasks),
                                          tx=optax.adamw(w_lr, weight_decay=w_d))
        aw_state = TrainState.create(apply_fn=None, params=jnp.zeros(num_tasks),
                                          tx=optax.adamw(w_lr, weight_decay=w_d))
        init_task_weights = jnp.ones((self.num_tasks,), dtype=jnp.float32) / jnp.maximum(self.num_tasks, 1)
        self.loss_process = loss_process
        self.models = Models(
            self.critic, self.actor, self.temp, self.target_critic, cw_state, aw_state,
            critic_loss=jnp.zeros(self.num_tasks),
            actor_loss=jnp.zeros(self.num_tasks),
            actor_grad_weights=init_task_weights,
            critic_grad_weights=init_task_weights,
        )
        self.warmup_epochs = warmup_epochs
        separate_grad_every = max(1, int(separate_grad_every))
        if grad_process_fn is None:
            grad_process_fn = resolve_grad_process_fn(grad_process)
        self.static_inputs = FrozenDict(
            {'discount': self.discount,
            'tau': self.tau,
            'target_entropy': self.target_entropy,
            'num_bins': self.num_bins,
            'v_max': self.v_max,
            'multitask': self.multitask,
            'loss_process': self.loss_process,
             'num_updates': updates_per_step,
             'warmup_done': False if warmup_epochs > 0 else True,
             'balance_critic': cfg.balance_critic,
             'use_separate_actor_grad': use_separate_actor_grad,
             'use_separate_critic_grad': use_separate_critic_grad,
             'separate_grad_every': separate_grad_every,
             'grad_process': grad_process,
             'grad_process_fn': grad_process_fn,
             'weight_interpolation_fn': default_weight_interpolation_fn,
             'alpha': grad_process_alpha,
             'lr': grad_lr,
            'momentum': grad_momentum,
                'niter': niter,
             }
        )
        self.cfg = cfg

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0):
        inputs = build_actor_input(self.critic, observations, self.task_ids, self.multitask)
        rng, actions = _sample_actions(self.rng, self.actor, inputs, temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def update(self, batch: Batch, num_updates: int, env_step: int):
        static_inputs = dict(self.static_inputs)
        should_run_separate_grad = (
            static_inputs['warmup_done']
            and (env_step % static_inputs['separate_grad_every'] == 0)
        )
        static_inputs['do_separate_actor_grad'] = bool(
            should_run_separate_grad and static_inputs['use_separate_actor_grad']
        )
        static_inputs['do_separate_critic_grad'] = bool(
            should_run_separate_grad and static_inputs['use_separate_critic_grad']
        )
        self.static_inputs = FrozenDict(static_inputs)

        step, rng, self.models, info = _do_multiple_updates(
            self.rng,
            self.models,
            batch,
            self.static_inputs,
            env_step
        )

        self.step = step
        self.rng = rng
        self.actor = self.models.actor
        self.critic = self.models.critic
        self.target_critic = self.models.target_critic
        self.temp = self.models.temp

        if self.step - self.cfg.start_training > self.warmup_epochs and not self.static_inputs['warmup_done']:
            new_static = dict(self.static_inputs)
            new_static['warmup_done'] = True
            self.static_inputs = FrozenDict(new_static)

            self.models = self.models.replace(
                actor_loss=info['actor_loss'],
                critic_loss=info['critic_loss'],
            )
        return info
    
    def get_infos(self, batch: Batch):
        # infos = _get_infos(
        #             self.rng,
        #             self.actor,
        #             self.critic,
        #             self.target_critic,
        #             self.temp,
        #             batch,
        #             self.discount,
        #             self.target_entropy,
        #             self.num_bins,
        #             self.v_max,
        #             self.multitask)
        # return infos
        *_, info = _do_multiple_updates(
            self.rng,
            self.models,
            batch,
            self.static_inputs,
            self.step,
        )
        return info
    
    def get_temperature(self):
        return _get_temperature(self.temp)

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.temp, self.rng = self.init_models(self.seeds)
        
    def save(self, path):
        self.actor.save(f'{path}/actor.txt')
        self.critic.save(f'{path}/critic.txt')
        self.target_critic.save(f'{path}/target_critic.txt')
        self.temp.save(f'{path}/temp.txt')
        
    def load(self, path):
        self.actor = self.actor.load(f'{path}/actor.txt')
        self.critic = self.critic.load(f'{path}/critic.txt')
        self.target_critic = self.target_critic.load(f'{path}/target_critic.txt')
        self.temp = self.temp.load(f'{path}/temp.txt')
        self.models = self.models.replace(
            actor=self.actor,
            critic=self.critic,
            target_critic=self.target_critic,
            temp=self.temp,
        )
