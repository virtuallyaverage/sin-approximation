import json

json_path = "E:\coding\AI\sin-approximation\sin\models\model_evals.json"

# Load the JSON data from a file
with open(json_path, 'r') as file:
    data = json.load(file)

# Sort the data by fitness, from highest to lowest
data['records'].sort(key=lambda x: x['fitness'], reverse=True)

# Save the sorted data back to the file
with open(json_path, 'w') as file:
    json.dump(data, file, indent=2)
