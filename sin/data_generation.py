import numpy as np
from torch.utils.data import Dataset
from torch import empty, from_numpy, is_tensor


class GenerateData():
    """Data generator class for dataset creation with various functions

    length_of_dataset = 10_000
    
    #initiallize the class
    data_generator = GenerateData()
    
    #setup how much data got generate
    data_generator.length_of_dataset = length_of_dataset
    torch_class = data_generator.get_torch_dataset()
    print(type(torch_class))
    
    Args:
        length_of_dataset (int, optional): the length of data to be generated. Needs to be defined before use but defaults to None.
    """
    def __init__(self, length_of_dataset = None) -> None:
        self.length_of_dataset = length_of_dataset
        self.verbose = False
        self.orignal_values = None
        self.output_values = None
        
        if self.verbose:
            print(f"using verbose mode, expect more print statements")
            
            if self.length_of_dataset == None:
                print("A dataset length needs to be set before use of generator functions")
        pass
    
    def get_torch_dataset(self):
        if self.length_of_dataset == None:
            print("no length provided")
            return
        
        #setup how much data got generate
        self.set_dataset_length(self.length_of_dataset)
        
        #actually calculate the values
        self.get_sin_and_value()
        
        #return the torch dataclass
        return self.create_torch_dataset()
        
    def get_sin_and_value(self, return_arrays:bool = True):
        """creates a random array, then applies the sin function to it. Both arrays are returned in a tuple (original, sin)
        Conditions:
            return_arrays(bool): if false will do operaiton in place with class variables. Useful for further class manipulation
        Returns:
            (conditionally) tuple of arrays: a tuple of arrays which is the original values and the values from applying sin to that array
        """
        #get orignal values
        self.orignal_values = np.random.rand(self.length_of_dataset)

        #get sin values
        self.output_values = np.sin(self.orignal_values)
        
        if self.verbose:
            #check correct values
            print(f"original values: {self.orignal_values[:5]}....x{len(self.orignal_values)}")
            print(f"sin values: {self.output_values[:5]}....x{len(self.output_values)}")
        
        #return the arrays if requested
        if return_arrays:
            return self.orignal_values, self.output_values
        else:
            return
    
    def set_dataset_length(self, dataset_length:int):
        """Sets the length of created datasets

        Args:
            dataset_length (int): desired length of datasets
        """
        self.length_of_dataset = dataset_length
        if self.verbose:
            print(f"dataset target length set to {dataset_length}")
            
    def create_torch_dataset(self):
        """reutrn a torch dataset that can be used in training a neural model

        Returns:
            _type_: _description_
        """
        #copy so that we can access them inside the dataset(sloppy)
        original_values = self.orignal_values
        function_values = self.output_values
        
        #setup dataset class to be returned
        class TwoValDataset(Dataset):
            def __init__(self) -> None:
                self.inputs = original_values
                self.outputs = function_values

                
            def __getitem__(self, index):
                if is_tensor(index):
                    index = index.tolist()
            
                #create tensors from values
                input_tensor = empty(self.inputs[index])
                output_tensor = empty(self.outputs[index])

                input_tensor.copy_(from_numpy(self.inputs))
                output_tensor.copy_(from_numpy(self.outputs))
                
                return input_tensor, output_tensor
            
            def __len__(self):
                return len(self.inputs)
            
        return TwoValDataset()
    
    
if __name__ == "__main__":
    length_of_dataset = 10_000
    
    #initiallize the class
    data_generator = GenerateData()
    
    #setup how much data got generate
    data_generator.length_of_dataset = length_of_dataset
    torch_class = data_generator.get_torch_dataset()
    print(type(torch_class))
    
