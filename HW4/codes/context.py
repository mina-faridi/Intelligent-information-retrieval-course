from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import csv
import re
import string
import pandas as pd
from sklearn.metrics import f1_score
import numpy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

datafile = open('news.csv', 'r')
myreader = csv.reader(datafile)

stop_words = set(stopwords.words('english'))
dicts = []
corpus = []
tags = [ "money-fx", "grain", "crude", "interest",  "earn", "trade","acq", "ship"]
y_true = []
for line in myreader:
    tokens = nltk.word_tokenize(line[1])
    filtered_sentence = [w for w in tokens if not w.lower() in stop_words]
    filtered_sentence = []

    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    words = [word for word in filtered_sentence if word.isalpha()]

    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]

    dict = {
        "id": line[0],
        "text": stemmed,
        "type": line[2]
    }
    dicts.append(dict)
    y_true.append(tags.index(line[2]))

    string_tokens=' '.join([str(elem) for elem in stemmed])
    corpus.append(string_tokens)

vectorizer = CountVectorizer()
term_occurrence = vectorizer.fit_transform(corpus).toarray()
terms = vectorizer.get_feature_names_out()

eowc_context=[[]]
for i in range(len(term_occurrence)):
    doc_size = sum(term_occurrence[i])
    for j in range(len(term_occurrence.T)):
        eowc_context[i][j] = term_occurrence.T[i][j]/len(doc_size)


bm25_context=[[]]
avdl = 0
b = 0.5
k = 5
for i in range (len (term_occurrence)):
    doc_size = sum(term_occurrence[i])
    avdl = avdl + doc_size
avdl = avdl / len(term_occurrence)
for i in range(len(term_occurrence)):
    doc_size = sum(term_occurrence[i])
    for j in range(len(term_occurrence.T)):
        bm25_context[i][j] = term_occurrence.T[i][j]*(k+1)/(term_occurrence.T[i][j]+k*(1-b+b*doc_size/avdl))
