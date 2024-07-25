import gymnasium as gym
import numpy as np
from PPO import PPOAgent

env = gym.make('MountainCar-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPOAgent(state_dim, action_dim)

rollout_len = 2048
memory = {'states': [], 'actions': [], 'rewards': [], 'action_probs': []}
total_rewards = []

for episode in range(1000):
    state = env.reset()[0]
    episode_reward = 0
    while True:
        action, action_prob = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)

        memory['states'].append(state)
        memory['actions'].append(action)
        memory['rewards'].append(reward)
        memory['action_probs'].append(action_prob)

        state = next_state
        episode_reward += reward

        if done:
            total_rewards.append(episode_reward)
            if len(memory['states']) >= rollout_len:
                agent.update(memory)
                memory = {'states': [], 'actions': [], 'rewards': [], 'action_probs': []}
            break

    if (episode + 1) % 10 == 0:
        print(f'Episode {episode+1}, Reward: {episode_reward}')