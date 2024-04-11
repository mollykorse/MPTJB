import json

file_path = r'C:\Users\Brandon\Documents\Datasets\2023.enriched.jsonl'

# Initialize a set to hold unique column names
columns = set()

# Read the first N lines to check for column names
N = 100  # Adjust N based on how many lines you want to sample
with open(file_path, 'r', encoding='utf-8') as file:
    for _ in range(N):
        line = file.readline()
        # Break if the end of file is reached
        if not line:
            break
        json_object = json.loads(line)
        columns.update(json_object.keys())

print("Unique column names found in the first {} lines:".format(N))
for column in columns:
    print(column)
