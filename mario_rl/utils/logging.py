import os
import torch

class Logger:
    def __init__(self, log_dir=None, use_wandb=True):
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                if log_dir is not None:
                    os.makedirs(log_dir, exist_ok=True)
                self.wandb.init(project="mario-rl", dir=log_dir)
                self._wandb_enabled = True
            except ImportError:
                self._wandb_enabled = False
                print("ERROR: wandB not installed. Metrics will not be stored")
        else:
            self._wandb_enabled = False

    def log(self, *args, **kwargs):
        if self._wandb_enabled:
            self.wandb.log(*args, **kwargs)

    def log_metrics(self, state, info, history):
        if self._wandb_enabled:
            self._log_wandb_metrics(state, info, history)

    def _log_wandb_metrics(self, state, info, history):
        if history["done"]:
            image_caption = "step: {} action: {}".format(
                history['current_step'], history['action'])

            self.wandb.log(
                {"frame": [self.wandb.Image(state, caption=image_caption)]}, 
                step=history['current_step'], 
                commit=False
            )

        self.wandb.log({**info, **history}, step=history['current_step'])

    def watch(self, model):
        if self._wandb_enabled:
            # Log model with wandb
            self.model = model
            self.wandb.watch(self.model)

    def close(self):
        if self._wandb_enabled:
            # Save model to wandb
            model_path = os.path.join(self.wandb.run.dir, 'model.pt')
            torch.save(self.model.state_dict(), model_path)
