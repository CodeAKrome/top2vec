
import json
import pandas as pd
from top2vec import Top2Vec

# Specify the path to the JSON file
file_path = '/home/kyle/src/mygit/top2vec/art.json'

# Initialize an empty list to store the JSON objects
json_list = []

# Open the file and read it line by line
with open(file_path, 'r') as file:
    for line in file:
        # Parse each line as a JSON object and append it to the list
        json_list.append(json.loads(line))

# Convert the list of JSON objects into a pandas DataFrame
df = pd.DataFrame(json_list)

# Convert all data in the DataFrame to string type and remove all newlines
df = df.applymap(lambda x: str(x).replace("\n", ''))

# Load the data into a Top2Vec model
model = Top2Vec(documents=df['text'].tolist(), speed='learn', workers=4)

# Print all the available functions in the Top2Vec model along with their documentation
print(dir(model))
