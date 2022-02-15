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


tfidf_vectorizer = TfidfVectorizer(max_features=3000)
tfidf = tfidf_vectorizer.fit_transform(corpus)

kmeans = KMeans(n_clusters=8).fit(tfidf)    

lines_for_predicting = corpus
y_pred = kmeans.predict(tfidf_vectorizer.transform(lines_for_predicting))

df = pd.DataFrame([y_true, y_pred])
f = open("df.csv", "w")
numpy.savetxt("df.csv",df, "%i")
f.close()

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # print(contingency_matrix)
    return numpy.sum(numpy.amax(contingency_matrix, axis=0)) / numpy.sum(contingency_matrix) 

print("Purity ", purity_score(y_true, y_pred))

print("RI     ", adjusted_rand_score(y_true, y_pred))

print("NMI    ", normalized_mutual_info_score(y_true, y_pred))

print("f1     ", f1_score(y_true, y_pred, average='weighted'))









