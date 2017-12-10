from __future__ import print_function

import codecs
import json
import re

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from scipy.cluster.hierarchy import ward, dendrogram
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

data_set = [json.loads(data) for data in codecs.open(filename="post_comments.json", encoding="utf-8").readlines()]
texts = [data['text'] for data in data_set]

# basic text preprocessing
stopwords = nltk.corpus.stopwords.words('russian')
stemmer = SnowballStemmer("russian")


def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Zа-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Zа-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens


words_stemmed = []
words_tokenized = []
for t in texts:
    words_stemmed.extend(tokenize_and_stem(t))
    words_tokenized.extend(tokenize_only(t))

vocab_frame = pd.DataFrame({'words': words_tokenized}, index=words_stemmed)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8,
                                   min_df=0.01, stop_words=stopwords,
                                   tokenizer=tokenize_and_stem)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_distances(tfidf_matrix)

# k cluster
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

print("Centers of clusters:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print()

linkage_matrix = ward(dist)
fig, ax = plt.subplots(figsize=(15, 20))
ax = dendrogram(linkage_matrix, orientation="right", labels=np.array([data['id'] for data in data_set]))
plt.show()
