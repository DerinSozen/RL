import gymnasium as gym
import numpy as np
from PPO import Agent
import pickle

if __name__ == '__main__':
    env = gym.make('highway-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape[0])
    n_games = 200

    best_score = env.reward_range[0]
    score_history = []

    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        j = 0
        while not done and j < 30000:
            j+= 1
            observation = np.array(observation).flatten().T
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done,_, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps)
    x = [i+1 for i in range(len(score_history))]
    
    pickle.dump(score_history, open('PPOrewards.pkl', 'wb'))