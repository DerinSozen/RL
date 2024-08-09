import pickle
import matplotlib.pyplot as plt

PPOsr = pickle.load(open("PPOsuccess_rates.pkl", "rb"))
HFPPOsr = pickle.load(open("HFPPOsuccess_rates.pkl", "rb"))
RSPPOsr = pickle.load(open("RSPPOsuccess_rates.pkl", "rb"))

plt.plot(PPOsr, label="PPO")
plt.plot(HFPPOsr, label="PPO with Reward Model")
plt.plot(RSPPOsr, label="PPO with Reward Model Reward Shaping")
plt.title("MountainCar Success Rates")
plt.xlabel("Episodes")
plt.ylabel("Success Rate")
plt.legend()
plt.savefig("MountainCarSuccessRates.png")
plt.clf()

PPOt = pickle.load(open("PPOtrain.pkl", "rb"))
HFPPOt = pickle.load(open("HFPPOtrain.pkl", "rb"))
RSPPOt = pickle.load(open("RSPPOtrain.pkl", "rb"))

plt.plot(PPOt, label="PPO")
plt.plot(HFPPOt, label="PPO with Reward Model")
plt.plot(RSPPOt, label="PPO with Reward Model Reward Shaping")
plt.title("MountainCar Training Rewards")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
plt.savefig("MountainCarTrainingRewards.png")




