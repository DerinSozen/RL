import gymnasium as gym
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
import pickle

obs_log = []
action_log = []
episode_rewards = []
episode_count = 0
episode_reward = 0

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global episode_count
    global episode_reward
    global log
    global episode_rewards
    episode_reward += rew
    obs_log.append(obs_t)
    action_log.append(action)
    if terminated or truncated:
        print(action)
        episode_count += 1
        episode_rewards.append(episode_reward)
        episode_reward = 0
    if episode_count >= 2:
        # When over episode count, raise exception to terminate play wrapper
        print(episode_count)
        raise Exception("Test Ended")


# Create the Highway Environment, set observation type to GrayscaleImage in config
env = gym.make("highway-v0",render_mode="rgb_array")
try:
    play(env, callback=callback,keys_to_action={'w':3, 's':4, 'a': 0, 'd': 1},fps=60, noop=1)
except Exception as e:
    pickle.dump(episode_rewards,open('play_rewards.pkl','wb'))
    pickle.dump(obs_log,open('play_observations.pkl','wb'))
    pickle.dump(action_log,open('play_actions.pkl','wb'))
    
    