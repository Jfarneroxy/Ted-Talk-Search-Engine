"""This Code will take search terms and desired number of results from the user and use TFIDF and cosine similarity to
provide the most relevant results in our dataset of 2467 TED Talks based on transcripts of those talks."""

import pandas as pd
import nltk
import numpy as np
import random
'''getting search terms and number of desired results WORKS'''
query_input = input("What would you like to hear a TED Talk on?: ")
num_results = int(input("How many results would you like? Results will be listed from most relevant to least. Enter integer value: "))

'''tokenize search terms WORKS'''
query_tokens =  nltk.word_tokenize(query_input)
query_tokens = [w.lower() for w in query_tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in query_tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
search_keys = [w for w in words if not w in stop_words]
#words = [w for w in words if not w in stop_words]
search_keys_array = np.array(search_keys)
#print(search_keys)
#print(search_keys_array)

data = search_keys
#data = [['meditation']]
queryframe = pd.DataFrame(data, columns = ['search_keys'])
#print(queryframe)
queryframe['search_keys'] = queryframe['search_keys'].apply(np.array)

'''loading in dataset WORKS'''
path = "/Users/jacobfarner/PycharmProjects/dc5/data.csv"
dataframe = pd.read_csv(path)

dataframe['transcript'] = dataframe['transcript'].apply(np.array)

#rint(dataframe)
'''transcript preprocessing WORKS BUT is handled through sklearn function so is unnecessary'''
'''
all_words = dataframe['transcript']
labeledData = []


#print(text.columns)
AllWords = []
text = ""
for i in all_words:
    labeledData.append((i))
    text+= i ;

random.shuffle(labeledData)
# take text containing all words and tokenize it
tokens =  nltk.word_tokenize(text)
tokens = [w.lower() for w in tokens]
# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
'''


'''calculating tfidf and search query weights WORKS'''
from sklearn.feature_extraction.text import TfidfVectorizer
label = "transcript"
label2 = "search_keys"
tfidf_vectorizer = TfidfVectorizer()
tfidf_weights = tfidf_vectorizer.fit_transform(dataframe.loc[:, label])
#tfidf_weights = tfidf_vectorizer.fit_transform(all_words)
search_query_weights = tfidf_vectorizer.transform(queryframe.loc[:, label2])
#print("Search Query Weights: ")
#print(search_query_weights)
#print("TFIDF Weights")
#print(tfidf_weights)


'''calculating cosine similarity between query vector and all other vectors WORKS'''
from sklearn.metrics.pairwise import cosine_similarity
cosine_distance = cosine_similarity(search_query_weights, tfidf_weights)
similarity_list = cosine_distance[0]


'''selecting ted talks with highest values WORKS'''
best_fit = []
while num_results > 0:
    tmp = np.argmax(similarity_list)
    best_fit.append(tmp)
    similarity_list[tmp] = 0
    num_results -=1

#print(best_fit)
'''returning list of TED Talk users WORKS'''
for i in best_fit:
    print(dataframe.iloc[i]['url'])





