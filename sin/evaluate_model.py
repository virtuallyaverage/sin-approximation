import torch
from tqdm import tqdm
import numpy as np
from math import sin, log
import torch.nn as nn
import os
import json
import hashlib



#path to model to load
models_folder = "E:\coding\AI\sin-approximation\sin\models"
model_name = "model_(9.79041715254425e-07).mdl"
model_path = os.path.join(models_folder, model_name)
json_name = "model_evals.json"

#the number to test values to evaluate, bigger=more accurate but longer to calculate
num_test = 10**4


#define the model that we are importing
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



# Generate input data and calculate sine values as output
print("creating data")
input_data = np.linspace(0, 2 * np.pi, num_test, dtype=np.float32)
np.random.shuffle(input_data)
output_data = np.array([sin(x) for x in input_data], dtype=np.float32)

#loading model
print(f"loading model at: {model_path}")
model = torch.load(model_path).cpu()


print("testing fitness")
total_difference = 0
total_percentage_difference = 0
for index in tqdm(range(0, len(input_data))):
    
    #get the test values for this itteration
    test_val, answer = input_data[index], output_data[index]
    
    #create tensors from data
    input_number = np.array(test_val, dtype=np.float32)
    input_number_tensor = torch.from_numpy(input_number.reshape(-1, 1)).cpu()
    
    #get the value
    output_number_tensor = model(input_number_tensor).cpu()
    output_number = output_number_tensor.detach().cpu().numpy()[0][0]
    
    #find the difference between the expected and the answer
    predicted_diff = abs(answer-output_number)
    total_difference += predicted_diff
    
    #find the percentage of inaccuracy.
    #if the answer is not zero, just continue
    if answer != 0:
        total_percentage_difference += predicted_diff/answer
        
#use data to calculate values
average = total_difference/num_test
model_fitness =  .1/average
average_percentage = total_percentage_difference/num_test
model_name = model_path.split("\\")[-1]

print("cleaning database")
#read in the json file
with open(os.path.join(models_folder, json_name), "r") as file:
    database = json.load(file)

print(f"Eval for: {model_name}")
print(f"The total difference across {num_test} values is {total_difference}")
print(f"Average deviation per value: {average}")
print(f"the model's fitness is {model_fitness}")
print(f"the average percentage {average_percentage}")


def hash_model_structure(model):
    model_str = str(model)
    model_bytes = model_str.encode('utf-8')
    hash_object = hashlib.sha256(model_bytes)
    hash_hex = hash_object.hexdigest()
    return hash_hex

def write_to_database(database, model):
    print("saving model")
    #make a unique model hash
    model_hash = hash_model_structure(model)
    
    # Create a new dictionary that represents the data you want to add
    new_entry = {
        "model_name": model_name,
        "total_value": num_test,
        "total_deviation": total_difference,
        "average": average,
        "fitness": model_fitness,
        "model_hash": model_hash,
        "average_percentage": average_percentage
    }
    
    # Append the new dictionary to the existing list of records
    database['records'].append(new_entry)

    # Save the modified data back to the file
    with open(os.path.join(models_folder, json_name), 'w') as file:
        json.dump(database, file, indent=2)
        
def remove_entry(database, model_name):
    for record in database['records']:
        if model_name == record['model_name']:
            print(f"deleted record {model_name}")
            database["records"].remove(record)
            return database
    print("entry not found")
    return database


# if the model is not in the database 
already_in_database = False
model_hash = hash_model_structure(model)
for record in database['records']:
    if model_name in record['model_name']:
        already_in_database = True
        break

#see if already in database
if not already_in_database:
    write_to_database(database, model)
        
#if already in database
else:
    should_overwrite = input("Already in database, overwrite? (Y/n)")
    if should_overwrite.lower() == 'y':
        database = remove_entry(database, model_name)
        write_to_database(database, model)
    else:
        print("skipping saving results")
        
