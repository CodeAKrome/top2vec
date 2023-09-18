#!env python
import json
import pandas as pd
from top2vec import Top2Vec

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

# Convert all data in the DataFrame to string type and remove all newlines
df = df.applymap(lambda x: str(x).replace("\n", ''))

# Print the first 5 rows of the DataFrame
print(df.head())

model = Top2Vec(df.text.tolist())
tpic_sz, tpic_n = model.get_topic_sizes()
# Basic stats
print(f"Samples: {len(df)}\ncats: {len(tpic_n)}")
print(f"Topic sizes:\n{tpic_sz}")

# words by topic
tpic_words, word_scores, tpic_n = model.get_topics()
for words, scores, n in zip(tpic_words, word_scores, tpic_n):
    print(f"{n}\nwords: {words}")

dox, dox_scores, dox_ids = model.search_documents_by_topic(topic_num=0, num_docs=10)
for doc, score, id in zip(dox, dox_scores, dox_ids):
    print(f"DocId: {id}, Score: {score}")
    print(doc)
    print("-"*8)
