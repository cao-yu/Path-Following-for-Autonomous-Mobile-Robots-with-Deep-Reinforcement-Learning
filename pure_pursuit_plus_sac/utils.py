# -*- coding: utf-8 -*-

import numpy as np
import torch

 
class ReplayBuffer:
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
	def __len__(self):
		return self.size

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)   



class NormalActionNoise:
    def __init__(self, mean, std_deviation):
        self.mean = mean
        self.std_dev = std_deviation

    def __call__(self):
        return np.random.normal(self.mean, self.std_dev, size=self.mean.shape)
    
    def reset(self):
        pass
    
    
    
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.       
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
            
        

class ActionScaler:
    def __init__(self, action_low, action_high):
        self.action_low = action_low
        self.action_high = action_high
        
    def forward(self, x): # [-1, 1] --> [low, high]
        if np.any(x < -1) or np.any(x > 1):
            raise ValueError("Invalid value")
            
        k = torch.FloatTensor(0.5 * (self.action_high - self.action_low))
        b = torch.FloatTensor(0.5 * (self.action_high + self.action_low))
        
        return k * x + b
        
    def inverse(self, x): # [low, high] --> [-1, 1]
        if np.any(x < self.action_low) or np.any(x > self.action_high):
            raise ValueError("Invalid value")
            
        k = torch.FloatTensor(2 / (self.action_high - self.action_low))
        b = torch.FloatTensor((self.action_high + self.action_low) / (self.action_high - self.action_low))
        
        return k * x - b
    