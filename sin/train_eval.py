import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_generation import GenerateData

#predefined variables
dataset_size = 10_000


#get the dataset
#initiallize the class
data_generator = GenerateData()

#setup how much data got generate
data_generator.length_of_dataset = dataset_size
input_data, output_data = data_generator.get_sin_and_value()

#convert to 32 bit floats
input_data = input_data.astype(np.float32)
output_data = output_data.astype(np.float32)

# Convert numpy arrays to PyTorch tensors
input_tensor = torch.from_numpy(input_data.reshape(-1, 1))
output_tensor = torch.from_numpy(output_data.reshape(-1, 1))

#define the model
class SinModel(nn.Module):
    def __init__(self) -> None:
        super(SinModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )
    
    #forward function
    def forward(self, x):
        return self.layers(x)
    
    
# Initialize the model, loss function, and optimizer
model = SinModel()#torch.load("models\model_(0.00013216079969424754)")#SinModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

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

    if (epoch+1) % 100 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
        
    if loss.item() < previous_loss:
        model_to_save = model
        previous_loss = loss.item()

# Test the model with a single input number
input_number = np.array([7], dtype=np.float32)
input_number_tensor = torch.from_numpy(input_number.reshape(-1, 1))
output_number_tensor = model(input_number_tensor)
output_number = output_number_tensor.detach().numpy()[0][0]

print(f'Input number: {input_number[0]}, Output number: {output_number}, expected value: {np.sin(input_number)}')

torch.save(model_to_save, f"E:\coding\AI\sin-approximation\sin\models\model_({loss.item()}).mdl")