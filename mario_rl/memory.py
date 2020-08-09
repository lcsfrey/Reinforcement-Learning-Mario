import random
import torch
import numpy as np

class ReplayMemory(object):
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.iteration = 0
    
    def push(self, state, next_state, action, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # add transision to memory
        # TODO: Remove samples from memory 
        #       inversely proportional to the prediction error
        self.memory[self.iteration] = [state, next_state, action, reward]

        self.iteration = (self.iteration + 1) % self.capacity
    
    def sample(self, batch_size=16):
        """ 
        Returns a four-tuple of
        batch_states, batch_next_states, batch_actions, batch_rewards
        """
        # sample random state transitions from memory
        sample = np.random.choice(len(self.memory), batch_size, replace=False)
         
        # access elements from memory according to sample
        samples = zip(*[self.memory[i] for i in sample])
        
        # return batched elements which can be differentiated
        return [
            torch.from_numpy(
                np.array(batch, 'float32')).requires_grad_(True) 
            for batch in samples
        ]

    def __len__(self):
        return len(self.memory)
