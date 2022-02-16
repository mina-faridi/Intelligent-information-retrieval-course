import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv('IMDB_Movie_Reviews.csv')

data['review'] = data['review'].str.strip().str.lower()
x = data['review']
y = data['sentiment'].astype(np.uint8)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, train_size = 0.8)

#vectorize by count
count_vectorizer= CountVectorizer(max_features=10000,ngram_range=(1,3),stop_words='english')
count_vectorizer.fit(x)
cv_train=count_vectorizer.transform(x_train)
cv_test= count_vectorizer.transform(x_test)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf= TfidfVectorizer(max_features=10000,ngram_range=(1,3),stop_words='english')
tfidf.fit(x)
tfidf_train=tfidf.transform(x_train)
tfidf_test= tfidf.transform(x_test)


# # Logistic Reg


#Logistic Regression with CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

lr= LogisticRegression(penalty='l1',solver='saga')
lr.fit(cv_train,y_train)
y_pred_lr_cv= lr.predict(cv_test)
print(confusion_matrix(y_test,y_pred_lr_cv))
print(classification_report(y_test,y_pred_lr_cv))
print(accuracy_score(y_test,y_pred_lr_cv))

#Logistic Regression with TFIDF Vectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

lr1= LogisticRegression(penalty='l2',solver='saga')
lr1.fit(tfidf_train,y_train)
y_pred_lr1_tfidf= lr1.predict(tfidf_test)
print(confusion_matrix(y_test,y_pred_lr1_tfidf))
print(classification_report(y_test,y_pred_lr1_tfidf))
print(accuracy_score(y_test,y_pred_lr1_tfidf))


# # svm


#Naive Bayes with CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

nv= MultinomialNB(alpha=0.1)
nv.fit(cv_train,y_train)
y_pred_nv_cv= nv.predict(cv_test)
print(confusion_matrix(y_test,y_pred_nv_cv))
print(classification_report(y_test,y_pred_nv_cv))
print(accuracy_score(y_test,y_pred_nv_cv))


#Naive Bayes with Tf-idf Vectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

nv1= MultinomialNB(alpha=0.1)
nv1.fit(tfidf_train,y_train)
y_pred_nv1_tfidf= nv.predict(cv_test)
print(confusion_matrix(y_test,y_pred_nv1_tfidf))
print(classification_report(y_test,y_pred_nv1_tfidf))
print(accuracy_score(y_test,y_pred_nv1_tfidf))


# # Naive Bayes


#Naive Bayes with CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

nv= MultinomialNB(alpha=0.1)
nv.fit(cv_train,y_train)
y_pred_nv_cv= nv.predict(cv_test)
print(confusion_matrix(y_test,y_pred_nv_cv))
print(classification_report(y_test,y_pred_nv_cv))
print(accuracy_score(y_test,y_pred_nv_cv))


#Naive Bayes with Tf-idf Vectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

nv1= MultinomialNB(alpha=0.1)
nv1.fit(tfidf_train,y_train)
y_pred_nv1_tfidf= nv.predict(cv_test)
print(confusion_matrix(y_test,y_pred_nv1_tfidf))
print(classification_report(y_test,y_pred_nv1_tfidf))
print(accuracy_score(y_test,y_pred_nv1_tfidf))


# # Logistic Reg


x_train_LR, x_test_LR, y_train_LR, y_test_LR = train_test_split(x,y, stratify=y, test_size=0.01, train_size = 0.04, random_state=42)


vec = CountVectorizer(stop_words='english')
x_train_LR = vec.fit_transform(x_train_LR).toarray()
x_test_LR = vec.transform(x_test_LR).toarray()


#Logistic Regression with CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

lr= LogisticRegression(penalty='l1',solver='saga')
lr.fit(x_train_LR,y_train_LR)
y_pred= lr.predict(x_test_LR)
print(confusion_matrix(y_test_LR,y_pred))
print(classification_report(y_test_LR,y_pred))
print(accuracy_score(y_test_LR,y_pred))


# # svm


x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(x,y, stratify=y, test_size=0.1, train_size = 0.4, random_state=42)

vec = CountVectorizer(stop_words='english')
x_train_svm = vec.fit_transform(x_train_svm).toarray()
x_test_svm = vec.transform(x_test_svm).toarray()



# Linear SVM with CountVectorizer

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

svm= LinearSVC(C=0.1)
svm.fit(x_train_svm,y_train_svm)
y_pred= svm.predict(x_test_svm)
print(confusion_matrix(y_test_svm,y_pred))
print(classification_report(y_test_svm,y_pred))
print(accuracy_score(y_test_svm,y_pred))


# # Naive Bayes


x_train_NB, x_test_NB, y_train_NB, y_test_NB = train_test_split(x,y, stratify=y, test_size=0.02, train_size = 0.08, random_state=42)


# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
x_train_NB = vec.fit_transform(x_train_NB).toarray()
x_test_NB = vec.transform(x_test_NB).toarray()

#Naive Bayes with CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

nv= MultinomialNB(alpha=0.1)
nv.fit(x_train_NB,y_train_NB)
y_pred= nv.predict(x_test_NB)
print(confusion_matrix(y_test_NB,y_pred))
print(classification_report(y_test_NB,y_pred))
print(accuracy_score(y_test_NB,y_pred))




