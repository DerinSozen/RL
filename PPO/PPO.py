import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Hyperparameters
learning_rate = 3e-4
gamma = 0.99
epsilon_clip = 0.2
k_epochs = 4
rollout_len = 2048

# Policy Network
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), action_probs[:, action.item()].item()

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        value = self.fc2(x)
        return value

# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = Policy(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.policy_old = Policy(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def update(self, memory):
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.LongTensor(memory['actions'])
        rewards = torch.FloatTensor(memory['rewards'])
        old_action_probs = torch.FloatTensor(memory['action_probs'])
        
        returns = []
        discounted_sum = 0
        for reward in reversed(rewards):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns)
        
        for _ in range(k_epochs):
            action_probs = self.policy(states)
            dist = Categorical(action_probs)
            new_action_probs = dist.log_prob(actions)

            ratios = torch.exp(new_action_probs - old_action_probs)
            advantages = returns - self.critic(states).squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-epsilon_clip, 1+epsilon_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            value_loss = nn.MSELoss()(self.critic(states).squeeze(), returns)
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def act(self, state):
        action, action_prob = self.policy_old.get_action(state)
        return action, action_prob
