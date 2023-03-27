import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
from os.path import exists, isdir, join
import random


num_epochs = 100000
model_path = "E:\coding\AI\sin-approximation\sin\models(1-128-64-1)(float)(1000step)\model_(9.391480944032082e-07).mdl" #set to None to make a new one
save_path = f"E:\coding\AI\sin-approximation\sin\models"
learning_rate = 0.001

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate input data and calculate sine values as output
input_data = np.linspace(0, 2 * np.pi, 2000000, dtype=np.float32)
np.random.shuffle(input_data)
output_data = np.array([math.sin(x) for x in input_data], dtype=np.float32)

# Convert numpy arrays to PyTorch tensors
input_tensor = torch.from_numpy(input_data.reshape(-1, 1)).to(device)
output_tensor = torch.from_numpy(output_data.reshape(-1, 1)).to(device)

# Define the neural network model
class SineModel(nn.Module):
    def __init__(self):
        super(SineModel, self).__init__()
        self.layer1 = nn.Linear(1, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

#load the model
if model_path != None:
    assert exists(model_path)
    model = SineModel().to(device)#torch.load(model_path).to(device)
    print(f"loading model {model_path}")
else: 
    model = SineModel().to(device)
    print("new model created")


# Initialize the model, loss function, and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Use a learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

# Train the model
previous_loss = 10*9
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_tensor)
    loss = criterion(outputs, output_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update the learning rate using the scheduler
    scheduler.step()

    if (epoch+1) % 500 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
        
    if loss.item() < previous_loss:
        model_to_save = model
        previous_loss = loss.item()
        best_loss = loss.item()

# Test the model with a single input number
input_number = np.array(.5, dtype=np.float32)
input_number_tensor = torch.from_numpy(input_number.reshape(-1, 1)).to(device)
output_number_tensor = model_to_save(input_number_tensor)
output_number = output_number_tensor.detach().cpu().numpy()[0][0]

print(f'Input number: {.5}, Predicted sine: {output_number}, Actual sine: {math.sin(.5)}')
print(f"the best loss: {best_loss}")
torch.save(model_to_save, join(save_path, f"model_({best_loss}).mdl"))