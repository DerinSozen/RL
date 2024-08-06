import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset = pickle.load(open('play_observations.pkl','rb'))
actions = pickle.load(open('play_actions.pkl','rb'))

print(actions)
print(np.array(dataset))

class RewardsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RewardsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
