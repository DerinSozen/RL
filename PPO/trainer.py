import gymnasium as gym
import numpy as np
from PPO import Agent
import pickle

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    eval_env = gym.make('MountainCar-v0')
    
    eval_rewards = []
    eval_success_rates = []
    
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 1000

    best_score = env.reward_range[0]
    score_history = []

    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        j = 0
        while not done:
            j+= 1
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, truncated , info = env.step(action)
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

        if i % 10 == 0:
            eval_reward = 0
            eval_success_rate = 0
            for k in range(10):
                observation, info = eval_env.reset()
                done = False
                truncated = False
                while not done and not truncated:
                    action, _, _ = agent.choose_action(observation)
                    observation, reward, done, truncated, _ = eval_env.step(action)
                    eval_reward += reward
                if done:
                    eval_success_rate += 1
            eval_reward /= 10
            eval_success_rate /= 10
            
            print('Evaluation: reward %.1f' % eval_reward, 'success rate %.1f' % eval_success_rate)
            
            eval_rewards.append(eval_reward)
            eval_success_rates.append(eval_success_rate)
        
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps)
    x = [j+1 for j in range(len(score_history))]
    
    pickle.dump(score_history, open('PPOtrain.pkl', 'wb'))
    pickle.dump(eval_rewards, open('PPOeval.pkl', 'wb'))
    pickle.dump(eval_success_rates, open('PPOsuccess_rates.pkl', 'wb'))