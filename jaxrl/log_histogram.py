import numpy as np
import wandb
import matplotlib.pyplot as plt


class WandbCriticCallback:
    def __init__(self, n_tasks, task_names, data_dim=101, max_history=500, attr_name="log_bins"):
        """
        data_dim: The number of outputs from your critic (width of the image).
        max_history: How many past steps to keep in the plot (height of the image).
        """
        super().__init__()
        self.data_dim = data_dim
        self.max_history = max_history

        # Initialize a buffer with NaNs (so empty spots show as white/blank)
        self.history_buffer = np.full((n_tasks, max_history,data_dim), np.nan)
        self.pos = 0
        self.task_names = task_names
        self.attr_name = attr_name

    def _on_step(self, current_values, step) -> bool:
        # self.history_buffer = np.roll(self.history_buffer, shift=-1, axis=1)
        safe_len = min(current_values.shape[-1], self.data_dim)
        self.history_buffer[:, self.pos, :safe_len] = current_values[:safe_len]
        self.pos += 1

        # 4. Log the image when full
        if self.pos == self.max_history:
            for i in range(self.history_buffer.shape[0]):
                self._plot_and_log(i, step)
            self.pos = 0

        return True

    def _plot_and_log(self, i, step):
        fig, ax = plt.subplots(figsize=(10, 6))

        # 'aspect=auto' allows the pixels to stretch to fill the box
        # 'interpolation=nearest' keeps pixels sharp (good for integer indices)
        im = ax.imshow(self.history_buffer[i, :self.pos].transpose(), cmap='viridis', aspect='auto', interpolation='nearest', origin='lower')

        # Add a colorbar to show magnitude
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Critic Value Magnitude')

        ax.set_title(f"{self.attr_name} (Step {step})")
        ax.set_ylabel("Bins")
        ax.set_xlabel("Timesteps")

        # Log to WandB
        if wandb.run is not None:
            wandb.log({f"{self.task_names[i]}/{self.attr_name}": wandb.Image(fig)}, step=step)

        plt.close(fig)