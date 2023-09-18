#!/usr/bin/env python3

import json
import pandas as pd

# Specify the path to the JSON file
file_path = '/home/kyle/src/mygit/top2vec/090607091694001099_1_art.json'

   # Initialize an empty list to store the JSON objects
   json_list = []

   # Open the file and read it line by line
   with open(file_path, 'r') as file:
       for line in file:
           # Parse each line as a JSON object and append it to the list
           json_list.append(json.loads(line))

   # Convert the list of JSON objects into a pandas DataFrame
   df = pd.DataFrame(json_list)

   # Print the first 5 rows of the DataFrame
   print(df.head())
