import torch
import numpy as np
import os
import argparse

default_model = "model_(2.7785529255197616e-06).mdl"

#get commandline arguments
parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", type=str, help="use specific model name", default=default_model)
args = parser.parse_args()

model_name = args.model

#path to model to load
model_name = args.model
models_folder = "sin\\models"
model_path = os.path.join(models_folder, model_name)

#get number from user
num_to_test = int(input(f"What number to take sin of? (0, 6.28)radians: "))

#define the model that we are importing
# Define the neural network model
class SineModel(torch.nn.Module):
    def __init__(self):
        super(SineModel, self).__init__()
        self.layer1 = torch.nn.Linear(1, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

#loading model
model = torch.load(model_path).cpu()

#create input tensor (make torch able to understand it)
input_number = np.array(num_to_test, dtype=np.float32)
input_number_tensor = torch.from_numpy(input_number.reshape(-1, 1)).cpu()

#get the value
output_number_tensor = model(input_number_tensor).cpu()
output_number = output_number_tensor.detach().cpu().numpy()[0][0]

print(f"Number predicted {output_number}")