# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Soft Actor-Critic (SAC)
# Paper: https://arxiv.org/abs/1801.01290
# Paper: https://arxiv.org/abs/1812.05905
# Newer:
# Value function is omitted
# Entropy regularization coefficient alpha is learnable


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(256, 256), action_space=None):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.mean = nn.Linear(hidden_size[1], action_dim)
        self.log_std = nn.Linear(hidden_size[1], action_dim)
        
        self.apply(weights_init_)
        
        if action_space is None:
            self.action_k = torch.tensor(1.)
            self.action_b = torch.tensor(0.)
        else:
            self.action_k = torch.FloatTensor(
                0.5 * (action_space.high - action_space.low))
            self.action_b = torch.FloatTensor(
                0.5 * (action_space.high + action_space.low))

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Limit the range of log_std for numerical stability
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick (mean + std * N(0,1))
        u = normal.rsample()
        a = torch.tanh(u)
        
        # Enforcing Action Bound
        action = self.action_k * a + self.action_b
        log_prob = normal.log_prob(u) - torch.log(self.action_k * (1 - a.pow(2)) + 1e-6)  # Log likelihood of the chosen action
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, self.action_k * torch.tanh(mean) + self.action_b
    
    def to(self, device):
        self.action_k = self.action_k.to(device)
        self.action_b = self.action_b.to(device)
        return super(Actor, self).to(device)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(256, 256)):
        super(Critic, self).__init__()

		# Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], 1)

		# Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l5 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l6 = nn.Linear(hidden_size[1], 1)
        
        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


class Agent(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        action_space,
        hidden_size=(256, 256),
        discount=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
    ):
 
        self.actor = Actor(state_dim, action_dim, action_space=action_space).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = - action_dim
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)

        self.discount = discount 
        self.tau = tau

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if not deterministic:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:      
            return
        
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
			# Select action according to policy
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            
			# Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha.detach() * next_log_prob
            target_Q = reward + not_done * self.discount * target_Q
            
  		# Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

  		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

  		# Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
		# Compute actor loss
        current_action, current_log_prob, _ = self.actor.sample(state)
        current_Q1, current_Q2 = self.critic(state, current_action)
        actor_loss = ((self.alpha * current_log_prob) - torch.min(current_Q1, current_Q2)).mean()
			
		# Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Compute temperature loss
        alpha_loss = -(self.log_alpha * (current_log_prob + self.target_entropy).detach()).mean()
        #alpha_loss = -(self.alpha * (current_log_prob + self.target_entropy).detach()).mean()
        
		# Optimize temperature 
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
		# Update the frozen target models
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau) 

    def soft_update(self, net_target, net, tau):
        for target_param, param  in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        #torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        #torch.save(self.critic.state_dict(), filename + "_critic")
        #torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        #self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        #self.actor_target = copy.deepcopy(self.actor)

        #self.critic.load_state_dict(torch.load(filename + "_critic"))
        #self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        #self.critic_target = copy.deepcopy(self.critic)
