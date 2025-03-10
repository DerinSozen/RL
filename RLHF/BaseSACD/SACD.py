import copy
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pickle
from torch.distributions.categorical import Categorical

class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Double_Q_Net, self).__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Identity()
        )
        self.Q2 = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Identity()
        )

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2

class Policy_Net(nn.Module):
    def __init__(self):
        super(Policy_Net, self).__init__()
        self.P = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Identity()
        )

    def forward(self, s):
        s = s.flatten()
        logits = self.P(s)
        probs = F.softmax(logits, dim=-1)
        return probs


class ReplayBuffer(object):
	def __init__(self, state_dim, dvc, max_size=int(1e6)):
		state_dim = 25
		self.max_size = max_size
		self.dvc = dvc
		self.ptr = 0
		self.size = 0

		self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.dvc)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.dvc)
		self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
		self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.dvc)

	def add(self, s, a, r, s_next, dw):
		self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
		self.a[self.ptr] = a
		self.r[self.ptr] = torch.from_numpy(np.array([r])).to(self.dvc)
		self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
		self.dw[self.ptr] = dw

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


class SACD_agent():
	def __init__(self,env):
		self.dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.state_dim = 25
		self.action_dim = env.action_space.n

		self.lr = 2e-4
		self.gamma = 0.99
		self.hid_shape = [128,128]
		self.batch_size = 1
		self.tau = 0.005
		self.H_mean = 0
		self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))
		self.alpha = 0.2

		self.actor = Policy_Net()
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

		self.q_critic = Double_Q_Net(self.state_dim,self.action_dim).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		for p in self.q_critic_target.parameters(): p.requires_grad = False


		self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))
		self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
		self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
			probs = self.actor(state)
			if deterministic:
				a = probs.argmax(-1).item()
			else:
				a = Categorical(probs).sample().item()
			return a

	def train(self):
		s, a, r, s_next, done = self.replay_buffer.sample(self.batch_size)

		# Train Critic Networks
		with torch.no_grad():
			next_probs = self.actor(s_next) 
			next_log_probs = torch.log(next_probs+1e-8) 
			next_q1_all, next_q2_all = self.q_critic_target(s_next) 
			min_next_q_all = torch.min(next_q1_all, next_q2_all)
			v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=-1, keepdim=True)
			target_Q = r + (~done) * self.gamma * v_next

		
		q1_all, q2_all = self.q_critic(s) 
		q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a) 
		q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()


		probs = self.actor(s) 
		log_probs = torch.log(probs + 1e-8) 
		with torch.no_grad():
			q1_all, q2_all = self.q_critic(s)
		min_q_all = torch.min(q1_all, q2_all)

		a_loss = torch.sum(probs * (self.alpha*log_probs - min_q_all), dim=-1, keepdim=True)

		self.actor_optimizer.zero_grad()
		a_loss.mean().backward()
		self.actor_optimizer.step()
  
		# Update Alpha
		with torch.no_grad():
			self.H_mean = -torch.sum(probs * log_probs, dim=-1).mean()
		alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

		self.alpha_optim.zero_grad()
		alpha_loss.backward()
		self.alpha_optim.step()
		self.alpha = self.log_alpha.exp().item()

		# Soft update target q networks using polyak averaging
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
