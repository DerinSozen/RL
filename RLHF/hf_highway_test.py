import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
import pickle

obs_log = np.empty((1,2))
action_log = []
episode_rewards = []
episode_count = 0
episode_reward = 0

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global obs_log
    global episode_count
    global episode_reward
    global log
    global episode_rewards
    episode_reward += rew
    print(np.array([obs_tp1]).shape)
    obs_log = np.vstack([obs_log,np.array(obs_tp1).T])
    action_log.append(action)
    print(np.array(obs_tp1))
    print(obs_log)
    if terminated or truncated:
        episode_count += 1
        episode_rewards.append(episode_reward)
        episode_reward = 0
    if episode_count >= 50:
        # When over episode count, raise exception to terminate play wrapper
        print(episode_count)
        raise Exception("Test Ended")


# Create the Highway Environment, set observation type to GrayscaleImage in config
env = gym.make("MountainCar-v0",render_mode="rgb_array")
try:
    play(env, callback=callback,keys_to_action={'w':0,'s':1,'d':2},fps=60, noop=1)
except Exception as e:
    print(e)
    print(obs_log)
    pickle.dump(obs_log,open('play_observations.pkl','wb'))
    pickle.dump(action_log,open('play_actions.pkl','wb'))
    
    