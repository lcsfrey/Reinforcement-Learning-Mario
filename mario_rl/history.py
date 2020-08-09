import numpy as np
from collections import defaultdict

class History(defaultdict):
    def __init__(self, logger=None, log_iters=None, num_previous_rewards=500, *args, **kwargs):
        super(History, self).__init__(list, *args, **kwargs)
        self.__dict__ = self
        self.num_previous_rewards = num_previous_rewards
        self.rewards_prev = np.zeros(num_previous_rewards)
        self.reward_next_idx = 0
        self.reward_recent_avg = 0.0
        self.reward_recent_min = 0.0
        self.reward_recent_max = 0.0
        self.logger = logger
        self.log_iters = log_iters
        self.logged_steps = []
        try:
            self.reward_best_avg
            self.reward_best_min
            self.reward_best_max
        except:
            self.reward_best_avg = 0.0
            self.reward_best_min = 0.0
            self.reward_best_max = 0.0

    def update(self, step_num=0, force_log=False, **kwargs):
        for k, v in kwargs.items():
            if k == 'reward':
                self.update_reward(v)

            self[k].append(v)

        if force_log or (self.log_iters is not None 
                         and (step_num % self.log_iters) == 0):
            self.logger.log_metrics(
                step=step_num, 
                prev_actions=self["action"][-self.num_previous_rewards:], 
                **kwargs
            )

    def update_reward(self, reward):
        # TODO: Move rewards to memory
        self.rewards_prev[self.reward_next_idx] = reward
        self.reward_next_idx = (self.reward_next_idx + 1) % self.num_previous_rewards
        self.reward_last_avg = self.reward_recent_avg
        self.reward_recent_avg = self.rewards_prev.mean()
        self.reward_recent_min = min(self.reward_recent_min, reward)
        self.reward_recent_max = max(self.reward_recent_max, reward)

        self.reward_best_avg = max(self.reward_best_avg, self.reward_recent_avg)
        self.reward_best_min = min(self.reward_best_min, self.reward_recent_min)
        self.reward_best_max = max(self.reward_best_max, self.reward_recent_max)

    @property
    def overall_avg(self):
        return self.reward_best_avg

    @property
    def recent_avg(self):
        return self.reward_recent_avg
    
    @property
    def overall_best(self):
        return self.reward_best_max

    @property
    def recent_best(self):
        return self.reward_recent_max

    def reset(self):
        self.__init__(logger=self.logger, log_iters=self.log_iters, 
                      num_previous_rewards=self.num_previous_rewards)