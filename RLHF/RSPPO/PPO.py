import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class Memory:
    def __init__(self, batch_size):
        # Initialize memory buffers
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        # Generate batches of memory for training
        n_states = len(self.states)
        batch_start_indices = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start_indices]
        
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, prob, val, reward, done):
        # Store experience in memory
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, learning_rate, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        
        # Define the network architecture
        self.actor = nn.Sequential(
            nn.Linear(input_dims[0], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Forward pass through the network
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        
        # Define the network architecture
        self.critic = nn.Sequential(
            nn.Linear(input_dims[0], fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Forward pass through the network
        value = self.critic(state)
        
        return value

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        
        # Initialize memory
        self.memory = Memory(self.batch_size)

    def remember(self, state, action, prob, val, reward, done):
        # Store experience in memory
        self.memory.store_memory(state, action, prob, val, reward, done)

    def choose_action(self, observation):
        # Choose action based on current policy
        state = T.tensor(np.array(observation), dtype=T.float).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        
        action = dist.sample()
        action_log_prob = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        
        return action, action_log_prob, value

    def learn(self):
        # Learning process
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, val_arr, reward_arr, done_arr, batches = self.memory.generate_batches()
            values = val_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            for t in range(len(reward_arr) - 1):
                discount = 1
                advantage_t = 0
                for k in range(t, len(reward_arr) - 1):
                    advantage_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = advantage_t
            
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + 0.5 * critic_loss
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        self.memory = Memory(self.batch_size)