import gymnasium as gym
import numpy as np

import os
import pickle

from jaxrl.utils import Batch

class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int):
        self.observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.actions = np.empty((num_tasks, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4
        self.num_tasks = num_tasks

    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.next_observations[:, self.insert_index] = next_observation
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, num_batches: int):
        indx = np.random.randint(self.size * self.num_tasks, size=(num_batches, batch_size))
        task_indx, sample_indx = np.divmod(indx, self.size)
        observations = self.observations[task_indx, sample_indx, :]
        actions = self.actions[task_indx, sample_indx, :]
        rewards = self.rewards[task_indx, sample_indx]
        masks = self.masks[task_indx, sample_indx]
        next_observations = self.next_observations[task_indx, sample_indx, :]
        return Batch(observations=observations,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     next_observations=next_observations,
                     task_ids=task_indx)

    def sample_task_batches(self, batch_size = 32):
        indxs = np.random.randint(self.size, size=batch_size)
        task_ids = np.zeros((self.num_tasks, batch_size), dtype=np.int32) + np.arange(self.num_tasks, dtype=np.int32)[:, None]
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     next_observations=self.next_observations[:, indxs],
                     task_ids=task_ids)

    def sample_equal_task_batches(self, batch_size, num_batches):
        per_task_batch_size = batch_size // self.num_tasks
        sampled_indices = np.random.randint(0, self.size, size=(num_batches, self.num_tasks, per_task_batch_size))
        task_indices = np.arange(self.num_tasks)[None, :, None]
        # indxs = np.random.randint(self.size, size=(num_batches, per_task_batch_size))
        task_ids = np.arange(self.num_tasks)[None, :, None].repeat(num_batches, axis=0).repeat(per_task_batch_size, axis=2)
        return Batch(observations=self.observations[task_indices, sampled_indices],
                     actions=self.actions[task_indices, sampled_indices],
                     rewards=self.rewards[task_indices, sampled_indices],
                     masks=self.masks[task_indices, sampled_indices],
                     next_observations=self.next_observations[task_indices, sampled_indices],
                     task_ids=task_ids)

        # indxs = np.random.randint(self.size, size=(self.num_tasks, per_task_batch_size))
        # task_ids = np.zeros((self.num_tasks, per_task_batch_size), dtype=np.int32) + np.arange(self.num_tasks, dtype=np.int32)[:, None]
        # return Batch(observations=np.take_along_axis(self.observations, indxs[..., None], axis=1),
        #              actions=np.take_along_axis(self.actions, indxs[..., None], axis=1),
        #              rewards=np.take_along_axis(self.rewards, indxs, axis=1),
        #              masks=np.take_along_axis(self.masks, indxs, axis=1),
        #              next_observations=np.take_along_axis(self.next_observations, indxs[..., None], axis=1),
        #              task_ids=task_ids)

    def sample_task_proportions(
        self,
        batch_size: int,
        num_batches: int,
        task_proportions: np.ndarray,
        min_samples_per_task: int = 0,
    ):
        """Sample fixed-size, flat batches with variable task composition.

        ``task_proportions`` may contain one distribution shared by every batch,
        or one distribution per batch. Counts are drawn independently for every
        batch, so the JAX-visible batch shape stays fixed while task counts vary.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0 or num_batches <= 0:
            raise ValueError("batch_size and num_batches must be positive.")
        if min_samples_per_task < 0:
            raise ValueError("min_samples_per_task must be non-negative.")

        reserved_samples = min_samples_per_task * self.num_tasks
        if reserved_samples > batch_size:
            raise ValueError(
                "min_samples_per_task * num_tasks cannot exceed batch_size."
            )

        proportions = np.asarray(task_proportions, dtype=np.float64)
        if proportions.ndim == 1:
            if proportions.shape != (self.num_tasks,):
                raise ValueError(
                    f"Expected {self.num_tasks} task proportions, got {proportions.shape}."
                )
            proportions = np.broadcast_to(proportions, (num_batches, self.num_tasks))
        elif proportions.shape != (num_batches, self.num_tasks):
            raise ValueError(
                "task_proportions must have shape (num_tasks,) or "
                f"(num_batches, num_tasks); got {proportions.shape}."
            )

        if not np.all(np.isfinite(proportions)) or np.any(proportions < 0):
            raise ValueError("task_proportions must be finite and non-negative.")
        row_sums = proportions.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Every task-proportion row must have a positive sum.")
        proportions = proportions / row_sums

        batch_task_ids = np.empty((num_batches, batch_size), dtype=np.int32)
        batch_sample_ids = np.empty((num_batches, batch_size), dtype=np.int64)
        remaining = batch_size - reserved_samples

        for batch_index in range(num_batches):
            counts = np.random.multinomial(remaining, proportions[batch_index])
            counts += min_samples_per_task
            if np.any(counts == 0):
                raise ValueError(
                    "Dynamic sampling requires every task to have at least one "
                    f"sample, but batch {batch_index} has counts {counts.tolist()}."
                )
            task_ids = np.repeat(
                np.arange(self.num_tasks, dtype=np.int32), counts
            )
            # Shuffle so downstream code cannot depend on samples being grouped by task.
            permutation = np.random.permutation(batch_size)
            batch_task_ids[batch_index] = task_ids[permutation]
            batch_sample_ids[batch_index] = np.random.randint(
                self.size, size=batch_size
            )[permutation]

        return Batch(
            observations=self.observations[batch_task_ids, batch_sample_ids],
            actions=self.actions[batch_task_ids, batch_sample_ids],
            rewards=self.rewards[batch_task_ids, batch_sample_ids],
            masks=self.masks[batch_task_ids, batch_sample_ids],
            next_observations=self.next_observations[
                batch_task_ids, batch_sample_ids
            ],
            task_ids=batch_task_ids,
        )

    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))
