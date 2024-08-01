import matplotlib.pyplot as plt
import numpy as np
import pickle

ppo_rewards = pickle.load(open("PPOrewards.pkl", "rb"))

plt.plot(np.array(ppo_rewards))
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("PPO Rewards on Highway environment")
plt.savefig("highwayPPO.png")

plt.cla()

play_rewards = pickle.load(open("play_rewards.pkl", "rb"))
plt.plot(np.array(play_rewards))
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Human Player Rewards on Highway environment")
plt.ylim(bottom=0)
plt.savefig("highwayHuman.png")
