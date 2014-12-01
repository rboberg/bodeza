# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:49:04 2014

@author: Ross
"""
# Modules
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn import metrics

# load JSON data
repo_dir = 'C:/Users/Ross/Documents/GitHub/bodeza'
data_dir = os.path.dirname(repo_dir) + '/bodeza_data'
fd = open(data_dir + '/yelp_data/yelp_academic_dataset_review.json')

import_limit = 100000

data_list = []
i = 0
for line in fd:
    line_data = line.strip()
    if line_data:
        data_list.append(json.loads(line))
        i += 1
        if i >= import_limit: break


fd.close()

train_size = 90000
test_size = 10000

###############################
### sklearn package test

# Set up tokenizer
# stemming makes it take longer
# I may want to check accuracy w/ and w/out
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r"\w+").tokenize
def ftokenize(text):
    tokens = tokenizer(text)
    return [stemmer.stem(w) for w in tokens]

#ftokenize = RegexpTokenizer(r"\w+").tokenize

# Get train data
train_docs = []
star_train = np.array([])
cool_train = np.array([])
funny_train = np.array([])
useful_train = np.array([])
for i in range(train_size):
    train_docs.append(data_list[i]['text'])
    star_train = np.append(star_train,data_list[i]['stars'])
    cool_train = np.append(cool_train,data_list[i]['votes']['cool'])
    funny_train = np.append(funny_train,data_list[i]['votes']['funny'])
    useful_train = np.append(useful_train,data_list[i]['votes']['useful'])

# Create term document matrix
count_v = CountVectorizer(tokenizer=ftokenize, stop_words='english')

td_train = count_v.fit_transform(train_docs)

# Create term document times inverse document frequency matrix
tfidf_transform = TfidfTransformer()
tfidf_train = tfidf_transform.fit_transform(td_train)


# Train the model
mnbc = MultinomialNB().fit(tfidf_train, star_train)

# Get testing data
test_docs = []
star_test = np.array([])
cool_test = np.array([])
funny_test = np.array([])
useful_test = np.array([])
for i in range(train_size,test_size+train_size):
    test_docs.append(data_list[i]['text'])
    star_test = np.append(star_test,data_list[i]['stars'])
    cool_test = np.append(cool_test,data_list[i]['votes']['cool'])
    funny_test = np.append(funny_test,data_list[i]['votes']['funny'])
    useful_test = np.append(useful_test,data_list[i]['votes']['useful'])

# Create a term document matrix
td_test = count_v.transform(test_docs)

# Create the occurence * inverse frequency matrix
tdidf_test = tfidf_transform.transform(td_test)

# Create predictions
mnbc_pred = mnbc.predict(tdidf_test)
#star_df['multinomial_nb'] = mnbc_pred

# Acccuracy
np.mean(star_test == mnbc_pred)

# Accuracy of NLTK Naive Bayes
#np.mean(star_test == star_df['model_max_prob'])

### Stochastic Gradient Descent Classifier
sgdc_loss = 'log' #'hinge
sgdc = SGDClassifier(loss=sgdc_loss, penalty='l2', alpha=1e-5, n_iter=50)
sgdc.fit(tfidf_train, star_train)

sgdc_pred = sgdc.predict(tdidf_test)
sgdc_pred


### Stochastic Gradient Descent Regression
sgdr_loss= 'huber'
sgdr = SGDRegressor(loss=sgdr_loss, penalty='l2', alpha=1e-5, n_iter=50)
sgdr.fit(tfidf_train, cool_train)

sgdr_pred = sgdr.predict(tdidf_test)
sgdr_pred

pd.ols(y=cool_test,x=pd.DataFrame(sgdr_pred))

# Acccuracy
np.mean(star_test == sgdc_pred)

### More advanced accuracy tests

if sgdc_loss == 'log':
    
    prob_df = pd.DataFrame(np.array(sgdc.predict_proba(tdidf_test)).round(2))
    prob_df['star_test'] = star_test

print(metrics.classification_report(star_test, mnbc_pred))
print(metrics.classification_report(star_test, sgdc_pred))

print(metrics.classification_report(star_test, np.array([5 for i in range(len(star_test))])))

print(metrics.confusion_matrix(star_test, mnbc_pred))
print(metrics.confusion_matrix(star_test, sgdc_pred))
mnbc_conf = metrics.confusion_matrix(star_test, mnbc_pred)
sgdc_conf = metrics.confusion_matrix(star_test, sgdc_pred)

svcm_pct_conf = (sgdc_conf.astype(float) / np.apply_along_axis(sum, 0, sgdc_conf)).round(2)
print(svcm_pct_conf)

print(metrics.confusion_matrix(star_test, np.array([1 for i in range(len(star_test))])))

###################
### sklearn w/ optimized parameters

mnbc_pipe = Pipeline([('vect', CountVectorizer(tokenizer=ftokenize, stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])

mnbc_pipe = mnbc_pipe.fit(train_docs, star_train)

sgdc_pipe = Pipeline([('vect', CountVectorizer(tokenizer=ftokenize, stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,n_iter=5))])

sgdc_pipe = sgdc_pipe.fit(train_docs, star_train)

parameters = {'vect__ngram_range': [(1,1), (1,2)],
              'clf__alpha': (1e-2, 1e-3)
              }

sgdc_gs = GridSearchCV(sgdc_pipe, parameters, n_jobs=-1)



