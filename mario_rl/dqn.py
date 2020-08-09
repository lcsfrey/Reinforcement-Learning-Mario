import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .memory import ReplayMemory
from .model import get_decision_network


REPEAT_ITERS_MEAN = 8.0   # Mean time spent applying an action
REPEAT_ITERS_STD = 2.0    # Standard deviation of time spent applying an action

class DQN(nn.Module):
    
    def __init__(self, 
                 input_size, 
                 nb_action, 
                 gamma, 
                 lrate, 
                 temperature=1e-2,
                 reload_previous_model=True, 
                 store_feature_maps=True, cuda=True):
        super(DQN, self).__init__()

        self.gamma = torch.tensor(gamma)
        # Temperature of the action space
        temperature = 1.0 if temperature is not None else temperature
        self.temperature = torch.tensor(temperature)
        
        self.cuda = cuda
        if cuda:
            self.temperature = self.temperature.cuda().half()
            self.gamma = self.gamma.cuda().half()

        # Create the decision network
        self.decision_network = get_decision_network(input_size, nb_action, cuda=cuda)

        #creating memory with memory class
        self.memory = ReplayMemory(3000)

        self.optimizer = optim.Adam(
            self.decision_network.parameters(), 
            lr=lrate,  # learning rate
            )

        self.done = True
        self.update_rate = 32
        self.update_count = 0

        self.action_rate = REPEAT_ITERS_MEAN
        self.action_count = 0

        self.action_rate_sampler = torch.distributions.normal.Normal(
            loc=REPEAT_ITERS_MEAN, scale=REPEAT_ITERS_STD)

        if reload_previous_model:
            self.load()

        self.feature_maps = {}
        if store_feature_maps:
            for name, module in self.decision_network.named_children():
                module.register_forward_hook(self._get_feature_maps(name))
    
    def _get_feature_maps(self, name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = tuple(o.detach().cpu() for o in output)
            else:
                output = output.detach().cpu()
            self.feature_maps[name] = output
        return hook

    def select_action(self, state):
        self.action_count += 1

        if self.done or self.action_count % self.action_rate == 0:
            with torch.no_grad():
                state = torch.from_numpy(state.astype('float32')).cuda().half()    
                # Q value depends on state
                # Temperature parameter T will be a positive number 
                # and the closer it is to zero the less sure 
                # the NN will when taking an action
                q = self.decision_network(state)
                probs = F.softmax((q*self.temperature), dim=1)
                #create a random draw from the probability distribution created from softmax
                action = probs.multinomial(1)
                self.last_action = action

                # pick a new amount of time to take the next action
                # TODO: Predict the amount of time needed to take the next action
                self.action_rate = (
                    self.action_rate_sampler
                    .sample()
                    .clamp_(min=1)
                    .round_()
                    .int()
                    .item()
                )
        else:
            action = self.last_action

        return action

    
    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        # TODO: Incorporate learned cost function via actor/critic model
        # TODO: Create subgoals to try to maximize global goals.
        #       If those subgoals ddon't improve upon global goals, 
        #       throw away and create new subgoals
        #       Add relavent subgoals to memory

        print("Learning...")
        if self.cuda:
            batch_state = batch_state.cuda()
            batch_next_state = batch_next_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()

        outputs = self.decision_network(batch_state)[:, batch_action.long()]

        next_outputs = self.decision_network(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        
        td_loss = F.smooth_l1_loss(outputs, target)

        self.optimizer.zero_grad()

        td_loss.backward()

		# Update the weights of the network
        self.optimizer.step()

        return td_loss.detach().item()
    
    def update(self, state, action, reward, next_state, done):

        self.memory.push(state, next_state, action, reward)

        self.update_count += 1

        if len(self.memory) > 128 \
                and ((self.update_count % self.update_rate == 0) or done):
            batch = self.memory.sample(64)
            loss = self.learn(*batch)
        else:
            loss = None
        
        self.done = done
        return loss
    
    def state_dict(self):
        return {
            'state_dict': self.decision_network.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }

    def save(self):
        torch.save(self.state_dict(), 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.decision_network.load_state_dict(
                checkpoint['state_dict'], strict=False)
            self.optimizer.load_state_dict(
                checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")