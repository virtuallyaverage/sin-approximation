import torch
from tqdm import tqdm
import numpy as np
from math import sin
import torch.nn as nn
import os
import json
import hashlib
from progressbar import progressbar

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

def evaluate_models(models_folder:str, num_test:int = 10**4, json_name:str = "model_evals.json"):
    print(f"starting on models in {models_folder}")
    
    # Generate input data and calculate sine values as output
    print("creating data")
    input_data = np.linspace(0, 2 * np.pi, num_test, dtype=np.float32)
    np.random.shuffle(input_data)
    output_data = np.array([sin(x) for x in input_data], dtype=np.float32)
    
    #get all file names
    model_name_list = []
    for dirpath, _, filenames in os.walk(models_folder):
        print(dirpath)
        #get model names
        for model_name in filenames:
            if model_name[-4:] == ".mdl":
                model_name_list.append(model_name)

    #process each model
    for model_name in progressbar(model_name_list):
        #create the model path
        model_path = os.path.join(dirpath, model_name)
        
        #try to work with the model
        try:
            evaluate_model(model_path, json_name, input_data, output_data)
        except:
            print(f"failed model {model_name}")
            break
        
            
            
def evaluate_model(model_path, json_name, input_data, output_data):
    #loading model
    model = torch.load(model_path).cpu()
    
    #seomthing
    num_test = len(input_data)
    
    #test the fitness
    total_difference = 0
    for index in range(0, len(input_data)):
        
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
        
    #use data to calculate values
    average = total_difference/num_test
    model_fitness =  .1/average
    model_name = model_path.split("\\")[-1]
    
    #read in the json file
    with open(os.path.join(models_folder, json_name), "r") as file:
        database = json.load(file)
        
    #print so that WE know
    print(f"Eval for: {model_name}")
    print(f"The total difference across {num_test} values is {total_difference}")
    print(f"Average deviation per value: {average}")
    print(f"the model's fitness is {model_fitness}")
    
    def hash_model_structure(model):
        """get the hash for the model

        Args:
            model (_type_): _description_

        Returns:
            _type_: _description_
        """
        
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
            "model_hash": model_hash
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
        database = remove_entry(database, model_name)
        write_to_database(database, model)
    return
    
    
    

if __name__ == "__main__":
    #path to model to load
    models_folder = "E:\coding\AI\sin-approximation\sin\models"

    evaluate_models(models_folder,)