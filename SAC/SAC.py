import copy
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pickle
from torch.distributions.normal import Normal

class Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Net, self).__init__()
        self.Q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a):
        return self.Q(torch.cat([s, a], dim=1))

class Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, s):
        x = self.fc(s)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, s):
        mean, std = self.forward(s)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

class SAC_agent():
    def __init__(self):
        self.dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 2
        self.action_dim = 1

        self.lr = 3e-4
        self.gamma = 0.99
        self.batch_size = 256
        self.tau = 0.005
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.dvc, max_size=int(1e6))
        self.alpha = 0.2

        self.actor = Policy_Net(self.state_dim, self.action_dim).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.q1_critic = Q_Net(self.state_dim, self.action_dim).to(self.dvc)
        self.q2_critic = Q_Net(self.state_dim, self.action_dim).to(self.dvc)
        self.q1_critic_optimizer = torch.optim.Adam(self.q1_critic.parameters(), lr=self.lr)
        self.q2_critic_optimizer = torch.optim.Adam(self.q2_critic.parameters(), lr=self.lr)
        
        self.q1_critic_target = copy.deepcopy(self.q1_critic)
        self.q2_critic_target = copy.deepcopy(self.q2_critic)
        
        for p in self.q1_critic_target.parameters(): p.requires_grad = False
        for p in self.q2_critic_target.parameters(): p.requires_grad = False

        self.target_entropy = -np.prod((self.action_dim,)).item()
        self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float, requires_grad=True, device=self.dvc)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def select_action(self, state, deterministic):
        state = torch.FloatTensor(state).to(self.dvc).unsqueeze(0)
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0] if not deterministic else torch.tanh(self.actor.forward(state)[0]).detach().cpu().numpy()[0]

    def train(self):
        s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_a, next_log_prob = self.actor.sample(s_next)
            q1_target = self.q1_critic_target(s_next, next_a)
            q2_target = self.q2_critic_target(s_next, next_a)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            target_Q = r + (~done) * self.gamma * min_q_target

        q1 = self.q1_critic(s, a)
        q2 = self.q2_critic(s, a)
        q1_loss = F.mse_loss(q1, target_Q)
        q2_loss = F.mse_loss(q2, target_Q)

        self.q1_critic_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_critic_optimizer.step()

        self.q2_critic_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_critic_optimizer.step()

        a, log_prob = self.actor.sample(s)
        q1 = self.q1_critic(s, a)
        q2 = self.q2_critic(s, a)
        min_q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        for param, target_param in zip(self.q1_critic.parameters(), self.q1_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2_critic.parameters(), self.q2_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)