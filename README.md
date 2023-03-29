# sin-approximation 
A small weekend project for estimating the sin of a value between 0-2pi radians
    
# Setup on windows
* Install torch ```pip install torch``` then restart powershell (open a new window)
* Clone repository ```cd install/location``` ```git clone https://github.com/virtuallyaverage/sin-approximation.git```
* Navigate to repository and run file ```cd install/location``` ```python get_a_sin.py```

# Models

You can use a different model from the models folder by putting ``-m "model name here"`` in front of the run command. An example using a more accurate, but less predictable model: ```python get_a_sin.py -m "model_(9.790426247491268e-07).mdl"```

* ```model_(2.7785529255197616e-06).mdl``` (default) with a mean deviation of 0.00249rad and an mean error of 0.3074% it is not the lowest deviation, but it is the lowest error percentage. This discrepency suggests that it is more consistant across the evaluated values, which I decided would be best as the default.

* ```model_(9.790426247491268e-07).mdl``` (experimental) This is a model that has been trained over 10 hours, while previous model was trained in 30 minutes. With a mean deviation of .00069rad and an mean error of 1.142% it has the lowest deviation, but loses out in the mean error. This discrepency suggests that it is less consistant across the evaluated values but more accurate overall, making it a decent model, but possibly with unexpected results on occasion.
