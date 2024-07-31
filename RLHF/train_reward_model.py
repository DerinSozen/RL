import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle

user_experiences = pickle.load(open('play_log.pkl','rb'))
plt.imshow(user_experiences[0][0][1][0])
plt.show()
print(user_experiences[0][0][0][0].shape)
