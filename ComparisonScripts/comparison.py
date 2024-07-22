import pickle
import matplotlib.pyplot as plt

q_learning = pickle.load(open('q_learning_eval_episodes.pkl', 'rb'))
SACD = pickle.load(open('SACD_eval_episodes.pkl', 'rb'))

plt.plot(*zip(*SACD), label='SACD')
plt.plot(*zip(*q_learning), label='Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Average Evaluation Reward')
plt.title('Comparison of SACD and Q-Learning on MountainCar')
plt.legend()
plt.savefig('comparison.png')