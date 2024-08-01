import matplotlib.pyplot as plt
import numpy as np
import pickle

ppo_train = pickle.load(open("PPO_train.pkl", "rb"))
ppo_eval = pickle.load(open("PPO_eval.pkl", "rb"))
ppo_success = pickle.load(open("PPO_success.pkl", "rb"))

sacd_train = pickle.load(open("SACD_train.pkl", "rb"))
sacd_eval = pickle.load(open("SACD_eval.pkl", "rb"))
sacd_success = pickle.load(open("SACD_success.pkl", "rb"))

plt.plot(ppo_train, label="PPO training")
plt.plot(sacd_train, label="SACD training")
plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Log Rewards")
plt.title("Training rewards between PPO and SACD")
plt.savefig("Training_rewards.png")
plt.clf()

x_scaled = np.arange(len(ppo_eval)) * 10
plt.plot(x_scaled,np.array(ppo_eval), label="PPO evaluation")
plt.plot(x_scaled,np.array(sacd_eval), label="SACD evaluation")
plt.title("Evaluation rewards between PPO and SACD")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.legend()
plt.savefig("Evaluation_rewards.png")
plt.clf()

plt.plot(x_scaled,np.array(ppo_success), label="success rates")
plt.plot(x_scaled,np.array(sacd_success), label="SACD success rates")
plt.title("Success rates between PPO and SACD")
plt.xlabel("Episodes")
plt.ylabel("Success rates")
plt.legend()
plt.savefig("Success_rates.png")