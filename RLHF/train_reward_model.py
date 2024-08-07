import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle

dataset = pickle.load(open('play_observations.pkl','rb'))
actions = pickle.load(open('play_actions.pkl','rb'))

class RewardsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RewardsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load dataset
dataset = pickle.load(open('play_observations.pkl','rb'))
actions = pickle.load(open('play_actions.pkl','rb'))

# Convert dataset to tensors
X = torch.tensor(dataset, dtype=torch.float32)
y = torch.tensor(actions, dtype=torch.long)

class RewardsModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RewardsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Hyperparameters
input_dim = 2
hidden_dim = 10
output_dim = 3
learning_rate = 0.001
num_epochs = 100

# Initialize model, loss function, and optimizer
model = RewardsModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Store loss
    losses.append(loss.item())
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the loss
plt.plot(range(num_epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()