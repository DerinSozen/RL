import gymnasium as gym
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
import pickle

log = []
episode_rewards = []
episode_count = 0
episode_reward = 0

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global episode_count
    global episode_reward
    global log
    global episode_rewards
    episode_reward += rew
    print(rew)
    if(obs_tp1[0].shape[0] != 0):
        log.append((obs_t, action, obs_tp1, rew, terminated, truncated, info))
    if terminated or truncated:
        episode_count += 1
        episode_rewards.append(episode_reward)
        episode_reward = 0
    if episode_count >= 25:
        # When over episode count, raise exception to terminate play wrapper
        print(episode_count)
        raise Exception("Test Ended")


# Create the Highway Environment, set observation type to GrayscaleImage in config
env = gym.make("highway-v0",render_mode="rgb_array")
# config = {
#        "observation": {
#            "type": "GrayscaleObservation",
#            "observation_shape": (128, 64),
#            "stack_size": 1,
#            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
#            "scaling": 1.75,
#        },
#        "policy_frequency": 2
#    }
# env.configure(config)
# Use the play utility to interact with the environment
try:
    play(env, callback=callback,keys_to_action={'w':3, 's':4, 'a': 0, 'd': 1},fps=15, noop=1)
except Exception as e:
    pickle.dump(episode_rewards,open('play_rewards.pkl','wb'))
    pickle.dump(log,open('play_log.pkl','wb'))
    
    