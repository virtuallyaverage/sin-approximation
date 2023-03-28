import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
from time import process_time
from os.path import exists, isdir, join
import random
from progressbar import progressbar
import keyboard


num_epochs = 100000
model_path = "E:\coding\AI\sin-approximation\sin\models\model_(1.5805346720298985e-06).mdl" #set to None to make a new one
save_path = "E:\coding\AI\sin-approximation\sin\models"
learning_rate = 0.0000001

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate input data and calculate sine values as output
print("creating data")
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
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    
    
#load the model
if model_path != None:
    assert exists(model_path)
    model = torch.load(model_path).to(device)
    print(f"loading model {model_path}")
else: 
    print("creating model")
    model = SineModel().to(device)
    print("new model created")



# Initialize the model, loss function, and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)



# Train the model
previous_loss = 10*9
num_epochs = 1000000
print("starting epochs")
start_time = process_time()
for epoch in progressbar(range(num_epochs)):
    # Forward pass
    outputs = model(input_tensor)
    loss = criterion(outputs, output_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    if (epoch+1) % 500 == 0:
        elapsed = process_time()-start_time
        print(f'\nEpoch: {epoch+1}, Best Loss: {loss.item()}, Epochs/second: {500/elapsed}')
        start_time = process_time()
        
        # Test the model with a single input number
        input_number = np.array(.5, dtype=np.float32)
        input_number_tensor = torch.from_numpy(input_number.reshape(-1, 1)).to(device)
        output_number_tensor = model_to_save(input_number_tensor)
        output_number = output_number_tensor.detach().cpu().numpy()[0][0]
        print(f'Input number: {.5}, Predicted sine: {output_number}, Actual sine: {math.sin(.5)}')
        print(f"diff: {abs(output_number-math.sin(.5))}")
        
        
    if loss.item() < previous_loss:
        model_to_save = model
        previous_loss = loss.item()
        best_loss = loss.item()

    # Check for 's' key press and save the model if pressed
    if keyboard.is_pressed('s'):
        print("Saving the model...")
        torch.save(model_to_save, join(save_path, f"model_({best_loss}).mdl"))
        print("Model saved.")


print("computing the average difference")
total_difference = 0
for index in progressbar(range(0, len(input_data), 100)):
    test_val, answer = input_data[index], output_data[index]
    input_number = np.array(test_val, dtype=np.float32)
    input_number_tensor = torch.from_numpy(input_number.reshape(-1, 1)).to(device)
    output_number_tensor = model_to_save(input_number_tensor)
    output_number = output_number_tensor.detach().cpu().numpy()[0][0]
    #find the difference between the expected and the answer
    predicted_diff = abs(answer-output_number)
    total_difference += predicted_diff
    
    
print(f"average deviation: {(total_difference/(len(input_data)/100))}")
    

torch.save(model_to_save, join(save_path, f"model_({best_loss}).mdl"))