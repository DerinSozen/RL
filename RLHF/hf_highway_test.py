import gymnasium as gym
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
import pickle

log = []
episode_count = 0

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global episode_count
    if(obs_tp1[0].shape[0] != 0):
        log.append((obs_t, action, obs_tp1, rew, terminated, truncated, info))
        print(obs_tp1[0])
    if terminated or truncated:
        episode_count += 1
    if episode_count >= 25:
        # When over episode count, raise exception to terminate play wrapper
        raise Exception("Test Ended")


# Create the Highwat Environment, set observation type to GrayscaleImage in config
env = gym.make("highway-v0",render_mode="rgb_array")
config = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (128, 64),
           "stack_size": 1,
           "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
           "scaling": 1.75,
       },
       "policy_frequency": 2
   }
env.configure(config)
# Use the play utility to interact with the environment
try:
    play(env, callback=callback,keys_to_action={'w':3, 's':4, 'a': 0, 'd': 1},fps=15, noop=1)
except Exception as e:
    pickle.dump(log,open('play_log.pkl','wb'))
    
    