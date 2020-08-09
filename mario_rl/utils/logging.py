import os
import torch
import torch.nn.functional as F


def process_feature_maps(state, feature_maps):
    processed_feature_maps = {}
    for f_name, f_maps in feature_maps.items():
        if not isinstance(f_maps, tuple):
            f_maps = [f_maps]
        for i, f_map in enumerate(f_maps):
            if f_map.ndim == 4 and f_map.shape[-1] == f_map.shape[-2]:
                if f_map.shape[0] > 1:
                    f_map = f_map[0][None]
                # only use features which are an image
                state = torch.from_numpy(state.astype(float))
                normalized_f_map = f_map.mean(1, keepdim=True) / f_map.max()
                f_map = F.interpolate(normalized_f_map, size=state.size()[:2])
                f_map = f_map.view(state.shape[0], state.shape[1], 1)
                processed_feature_maps[f"activation.{f_name}.{i}"] = f_map * 255
                processed_feature_maps[f"attention.{f_name}.{i}"] = state*f_map
    return processed_feature_maps
class Logger:
    def __init__(self, env, model=None, log_dir=None, use_wandb=True):
        self.model = model
        self.num_actions = env.action_space.n
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

    def log_metrics(self, step=None, *args, **kwargs):
        if self._wandb_enabled:
            print("Logging to WandB")
            self._log_wandb_metrics(step=step, *args, **kwargs)

    def _log_wandb_metrics(self, step=None, *args, **kwargs):
        """
        if "state" in kwargs:
            image_caption = "step: {} action: {}".format(
                step, kwargs['action'])
            self.wandb.log(
                {"frame": [self.wandb.Image(kwargs["state"], caption=image_caption)]}, 
                step=step, 
                commit=False
            )

            if "feature_maps" in kwargs:
                feature_maps = process_feature_maps(kwargs["state"], 
                                                    kwargs["feature_maps"])

                kwargs["feature_maps"] = [
                    self.wandb.Image(f_map.numpy(), caption=f_name)
                    for f_name, f_map in feature_maps.items()
                ]
        """
        kwargs["prev_actions"] = self.wandb.Histogram(
            kwargs["prev_actions"], num_bins=self.num_actions)
        self.wandb.log(kwargs, step=step)

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
