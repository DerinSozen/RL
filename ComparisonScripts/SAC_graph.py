import pickle
import matplotlib.pyplot as plt

SAC = pickle.load(open('SAC_eval_episodes.pkl', 'rb'))

plt.plot(*zip(*SAC), label='SACD')
plt.xlabel('Episodes')
plt.ylabel('Average Evaluation Reward')
plt.title('SAC Eval episodes')
plt.legend()
plt.savefig('SAC_eval.png')