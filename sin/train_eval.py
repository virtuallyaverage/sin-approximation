import os
import numpy as np
import torch
from time import process_time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from data_generation import GenerateData

#predefined variables
dataset_size = 10_000


#get the dataset
#initiallize the class
data_generator = GenerateData()

#setup how much data got generate
data_generator.length_of_dataset = dataset_size
dataset = data_generator.get_torch_dataset()


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
        x = x.view(-1, 40000)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(f"Output after layer {i}:", x)
        return x
    
    
#divide the dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_set, batch_size=20, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=20, shuffle=False, num_workers=8)

#Define initial weights process
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
#a main fucntion to make torch happy   
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SinModel()
    # Initialize model, loss function, and optimizer
    model.apply(init_weights)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    accumulation_steps = 6
    scaler = GradScaler()
    # Training loop
    def train(model, dataloader, criterion, optimizer, scaler, device, accumulation_steps):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()
        return running_loss / len(dataloader)
    
    # Evaluation loop
    def evaluate(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
        #print("Inputs:", inputs)
        #print("Targets:", targets)
        #print("Outputs:", outputs)
        return running_loss / len(dataloader)

    # Train and evaluate the model
    num_epochs = 50
    for epoch in range(num_epochs):
        #print(f"starting epoch {epoch+1}, {num_epochs-epoch} to go")
        train_loss = train(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps=6)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    print("Training and evaluation complete.")
    print("saving model")
    # Save the model
    model_path = os.path.join("models", f"model_{num_epochs, len(model.layers), train_loss, val_loss}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

#run if not imported  
if __name__ == '__main__':
    big_start = process_time()
    main()
    print(f"completed in {(process_time()-big_start)/60} minutes")