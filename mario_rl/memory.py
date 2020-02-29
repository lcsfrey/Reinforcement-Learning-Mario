import random
import torch
import numpy as np

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity #100 transitions
        self.memory = [] #memory to save transitions
        self.iteration = 0
    
    def push(self, event):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # add transision to memory
        self.memory[self.iteration] = event
        self.iteration = (self.iteration + 1) % self.capacity
    
    # taking random sample
    def sample(self, batch_size):
        #Creating variable that will contain the samples of memory
        #zip =reshape function if list = ((1,2,3),(4,5,6)) zip(*list)= (1,4),(2,5),(3,6)
        #                (state,action,reward),(state,action,reward)  
        samples = zip(*random.sample(self.memory, batch_size))
        #This is to be able to differentiate with respect to a tensor
        #and this will then contain the tensor and gradient
        #so for state, action and reward we will store the seperately into some
        #bytes which each one will get a gradient
        #so that eventually we'll be able to differentiate each one of them


        #next_state = torch.from_numpy(next_state.astype('float32')).unsqueeze(0)
        #reward = torch.as_tensor(reward).float()
        #new_memory = (
        #    torch.from_numpy(state.astype('float32')).unsqueeze(0), 
        #    next_state, 
        #    torch.FloatTensor([int(action)]), 
        #    reward.unsqueeze(0)
        #)
        def stack(batch):
            return torch.from_numpy(np.array(batch, dtype='float32')).requires_grad_(True)
            
        return map(lambda batch: stack(batch), samples)

    def __len__(self):
        return len(self.memory)
