import pickle
import matplotlib.pyplot as plt
import numpy as np

q_learning = pickle.load(open('q_learning_eval_episodes.pkl', 'rb'))
SACD = pickle.load(open('SACD_eval_episodes.pkl', 'rb'))
PPO = pickle.load(open('PPOrewards.pkl', 'rb'))


# plt.plot(np.array(SACD)[:300], label='SACD')
# plt.plot(np.array(q_learning)[:300], label='Q-Learning')
plt.plot(np.array(PPO), label='PPO')
plt.xlabel('Episodes')
plt.ylabel('Average Training Reward')
plt.title('PPO performance on MountainCar')
plt.legend()
plt.savefig('PPO_training.png')