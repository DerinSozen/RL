from datetime import datetime
from SAC import SAC_agent
import gymnasium as gym
import os, shutil
import argparse
import pickle
import torch

#define hyperparameters
RANDOM_EPS = 100
UPDATE_EVERY = 5

def main():
	#Create Env
	env = gym.make("MountainCarContinuous-v0")
	eval_env = gym.make("MountainCarContinuous-v0")

	# Seed Everything
	env_seed = 0
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


	agent = SAC_agent()
	eps = 0
	total_steps = 0
	eval_episodes = []
	while eps < 100:
		eps += 1
		s, info = env.reset(seed=env_seed)
		env_seed += 1
		done = False
		truncated = False
		i = 0
		while not done:
			i+=1
			if eps < 50: a = env.action_space.sample()
			else: a = agent.select_action(s, deterministic=False)
			s_next, r, done, truncated, info = env.step(a)

			agent.replay_buffer.add(s, a, r, s_next, done)
			s = s_next

			if eps >= 50 and total_steps % UPDATE_EVERY == 0:
				for j in range(UPDATE_EVERY):
					agent.train()
			total_steps +=1
		if eps % 10 == 0:
			eval_rewards = 0
			for j in range(5):
				s, info = eval_env.reset()
				done = False
				truncated = False
				while not done and not truncated:
					a = agent.select_action(s, deterministic=True)
					s, r, done, truncated, info = eval_env.step(a)
					eval_rewards += r
			eval_episodes.append((eps,(eval_rewards/5)))

		print(f"Episode {eps} finished in {i} steps.")
			
	env.close()
	eval_env.close()

	pickle.dump(eval_episodes, open("SAC_eval_episodes.pkl", "wb"))

if __name__ == '__main__':
	main()

