import numpy as np

class History(dict):
    def __init__(self, num_previous_rewards=50, *args, **kwargs):
        super(History, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self.num_previous_rewards = num_previous_rewards
        self.rewards_prev = np.zeros(num_previous_rewards)
        self.reward_next_idx = 0
        self.reward_recent_avg = float('-inf')
        self.reward_recent_min = float('inf')
        self.reward_recent_max = float('-inf')
        try:
            self.reward_best_avg
            self.reward_best_min
            self.reward_best_max
        except:
            self.reward_best_avg = float('-inf')
            self.reward_best_min = float('inf')
            self.reward_best_max = float('-inf')

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'reward':
                self.update_reward(v)
            self[k] = v

    def update_reward(self, reward):
        self.last_reward = reward
        self.reward = reward

        self.rewards_prev[self.reward_next_idx] = np.array(reward)
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
        self.__init__()