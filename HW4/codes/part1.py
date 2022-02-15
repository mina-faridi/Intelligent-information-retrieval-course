import nltk
import csv
import numpy
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


datafile = open('news.csv', 'r')
myreader = csv.reader(datafile)
stop_words = set(stopwords.words('english'))

dicts = []
corpus = []
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

    string_tokens=' '.join([str(elem) for elem in stemmed])
    corpus.append(string_tokens)

vectorizer = CountVectorizer()
term_occurrence = vectorizer.fit_transform(corpus).toarray()
terms = vectorizer.get_feature_names_out()

iran = numpy.where(terms == "iran")
print(iran) #6521

# print(teacher)#12549



ttdm = numpy.transpose(term_occurrence)
mi = []
for j in range(0, 14122):
    mi.append(round(sklearn.metrics.mutual_info_score(ttdm[6521], ttdm[j]), 5))
iran_co = numpy.argpartition(mi, -10)[-10:]
print(iran_co)

maxt = mi
maxt.sort()
print(maxt[-10:])

for t in iran_co:
    print(mi[t])
    print(terms[t])