#!env python
import json
import pandas as pd
from top2vec import Top2Vec
import sys

# Naughty bits
#
#
text_field = "text"
bees = 16

# Yesterday's jam
# Specify the path to the JSON file
#file_path = '/home/kyle/src/mygit/top2vec/090607091694001099_1_art.json'

# Initialize an empty list to store the JSON objects
json_list = []

# Open the file and read it line by line
#with open(file_path, 'r') as file:
#    for line in file:
        # Parse each line as a JSON object and append it to the list
#        json_list.append(json.loads(line))


# We now return you to your regularly scheduled program

for line in sys.stdin:
   json_list.append(json.loads(line))


# Convert the list of JSON objects into a pandas DataFrame
df = pd.DataFrame(json_list)

# Convert all data in the DataFrame to string type and remove all newlines
df = df.applymap(lambda x: str(x).replace("\n", ''))

# Print the first 5 rows of the DataFrame
print(df.head())

# Phase 0
#model = Top2Vec(df.text.tolist())
model = Top2Vec(df[text_field].tolist(), speed="learn", workers=bees)

tpic_sz, tpic_n = model.get_topic_sizes()
tpic_cnt = len(tpic_n)
samples = len(df)
bar = "-"*8

# Basic stats
print(f"Embettering: Learning mode test bees={bees}\n{bar}\nMeat\nSamples: {samples}\ncats: {tpic_cnt}")
print(f"Topic sizes:\n{tpic_sz}")

print("Words by all topics\nget_topics()\ncount word cosine")
tpic_words, word_scores, tpic_n = model.get_topics()
for word, score, n in zip(tpic_words, word_scores, tpic_n):
    print(f"{n}\nword: {word} {score}")
print("-"*8)

print("Topic of Cancer\nPer topic documents | head -n 10\nsearch_documents_by_topic(topic_num=0, num_docs=10)")
dox, dox_scores, dox_ids = model.search_documents_by_topic(topic_num=0, num_docs=10)
for doc, score, id in zip(dox, dox_scores, dox_ids):
    print(f"DocId: {id}, Score: {score}")
    print(doc)
    print("-"*8)

# Search
# topic_words: For each topic the top 50 words are returned, in order of semantic similarity to topic.
# word_scores: For each topic the cosine similarity scores of the top 50 words to the topic are returned.
# topic_scores: For each topic the cosine similarity to the search keywords will be returned.
# topic_nums: The unique index of every topic will be returned.
print("search_topics(keywords=['ukraine'], num_topics=2)\nindex, word, word cos, topic cos")
tpix_words, words_cos, tpix_cos, tpix_idx = model.search_topics(keywords=["ukraine"], num_topics=2)
for tpic_word, word_cos, tpic_cos, tpic_idx in zip (tpix_words, words_cos, tpix_cos, tpix_idx):
    print(f"[{tpic_idx}] word: {tpic_word} {word_cos} topic: {tpic_cos}")

# Cloudy Weather
#for topic in tpix_idx:
#    model.generate_topic_wordcloud(topic)


# Similar keywords
print(f"Similar_keywords(ukraine), can have negative kwords num=13\nword cos")
words, words_cos = model.similar_words(keywords=["ukraine"], keywords_neg=[], num_words=13)
for word, wcos in zip(words, words_cos):
    print(f"{word} {wcos}")
