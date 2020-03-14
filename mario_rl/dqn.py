# Importing the libraries

import random # random samples from different batches (experience replay)
import os # For loading and saving brain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # for using stochastic gradient descent

from .memory import ReplayMemory
from .model import get_decision_network


REPEAT_ITERS_MEAN = 8.0         # Mean time spent applying an action
REPEAT_ITERS_STD = 2.0          # Standard deviation of time spent applying an action


def compute_cost(env_reward, x_pos, prev_x_pos=None, y_pos=None,  
                 prev_y_pos=None, time_take=1, done=False, overall_reward=0.0, 
                 discount_factor = 0.95):
    movement = 0
    if done:
        reward = -15
        overall_reward = 0
    elif prev_x_pos is not None:
        x_movement = torch.tensor(x_pos - prev_x_pos, dtype=torch.float32)

        #reward = movement * abs(env_reward)
        #reward = -1 if movement == 0 else reward

        reward = x_movement.exp() / time_take

        if y_pos is not None and prev_y_pos is not None:
            max_jump_height = 150
            ground_height = 78
            y_movement = (y_pos - ground_height)/max_jump_height
            y_movement = torch.tensor(y_movement, dtype=torch.float32)
            reward *= y_movement
    else:
        reward = env_reward
    reward = torch.tensor(reward)
    overall_reward = torch.tensor(overall_reward)
    discount_factor = torch.tensor(discount_factor)

    
    current_reward = reward
    overall_reward = reward + overall_reward * discount_factor
    return current_reward, overall_reward, movement

class DQN(nn.Module):
    
    def __init__(self, input_size, nb_action, gamma, lrate, history, temperature=None, reload_previous_model=True):
        super(DQN, self).__init__() #inorder to use modules in torch.nn
        self.gamma = torch.tensor(gamma).cuda().half() #self.gamma gets assigned to input argument
        # Temperature of the action space
        if temperature is None:
            temperature = torch.ones(nb_action).cuda().half()
        self.temperature = temperature

        self.history = history
        # Sliding window of the evolving mean of the last 100 events/transitions
        num_max_rewards = 1000
        self.reward_window = [torch.zeros(1) 
                              for _ in range(num_max_rewards)]
        #Creating network with network class
        self.decision_network = get_decision_network(input_size, nb_action).cuda()  
        #self.decision_network = torch.jit.script(self.decision_network)
        #creating memory with memory class
        #We gonna take 100000 samples into memory and then we will sample from this memory to 
        #to get a number of random transitions
        self.memory = ReplayMemory(500)
        #creating optimizer (stochastic gradient descent)
        self.optimizer = optim.Adam(self.decision_network.parameters(), lr=lrate) #learning rate

        self.done = True
        self.update_rate = 10
        self.update_count = 0

        self.action_rate = 10
        self.action_count = 0

        
        self.action_rate_sampler = torch.distributions.normal.Normal(
            loc=REPEAT_ITERS_MEAN, scale=REPEAT_ITERS_STD)

        if reload_previous_model:
            self.load()
    
    def select_action(self, state):
        if self.done or self.action_count % self.action_rate == 0:
            with torch.no_grad():
                state = torch.from_numpy(state.astype('float32')).cuda().half()    
                #Q value depends on state
                #Temperature parameter T will be a positive number and the closer
                #it is to zero the less sure the NN will when taking an action
                #forexample
                #softmax((1,2,3))={0.04,0.11,0.85} ==> softmax((1,2,3)*3)={0,0.02,0.98} 
                #to deactivate brain then set T=0, thereby it is full random
                unnormalized_probs = self.decision_network(state)
                probs = F.softmax((unnormalized_probs*self.temperature), dim=1) # T=100
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

        self.action_count += 1
        
        return action.data[0,0]

    @property
    def hidden_state(self):
        return self.decision_network.lstm_state
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.decision_network(batch_state.cuda().half()).gather(1, batch_action.cuda().long().unsqueeze(1)).squeeze(1)
        #next input for target see page 7 in attached AI handbook
        next_outputs = self.decision_network(batch_next_state.cuda().half()).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward.cuda()
        #Using hubble loss inorder to obtain loss
        td_loss = F.smooth_l1_loss(outputs, target)
        #using  lass loss/error to perform stochastic gradient descent and update weights 
        self.optimizer.zero_grad() #reintialize the optimizer at each iteration of the loop
        #This line of code that backward propagates the error into the NN
        td_loss.backward(retain_graph=True)
		#And this line of code uses the optimizer to update the weights
        self.optimizer.step()
    
    def update(self, state, action, reward, next_state, done):
        #Updated one transition and we have dated the last element of the transition
        #which is the new state
        self.memory.push((state, next_state, action, reward))

        self.update_count += 1

        alpha = torch.empty(1).normal_(mean=1.3,std=0.1)

        improving_recently = self.history.recent_avg > self.history.reward_last_avg * alpha
        improving_globally = self.history.recent_avg > self.history.reward_best_avg
        """
        if len(self.memory) > 100 \
                and (self.update_count + 1) % self.update_rate == 0:
            print(len(self.memory) > 100)
            print((self.update_count + 1) % self.update_rate == 0)
            print(self.history.recent_avg)
            print( self.history.reward_last_avg*1.15)
            print(improving_recently or improving_globally or done)
            print("-------------")
        """
        if len(self.memory) > 100 \
                and (self.update_count + 1) > self.update_rate \
                and (improving_recently or improving_globally or done):
            if improving_globally:
                print("Improving globally")
            if improving_recently:
                print("Improving recently")
            if done:
                print("Done")
            print("------------------------")
            batch_state, batch_next_state, batch_action, batch_reward \
                = self.memory.sample(50)
            self.learn(batch_state, batch_next_state, 
                       batch_reward, batch_action)
            self.update_count = 0
        self.reward_window[self.update_count % len(self.reward_window)] = reward

        self.done = done
        return action
    
    def score(self):
        return self.reward_window.mean()
    
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