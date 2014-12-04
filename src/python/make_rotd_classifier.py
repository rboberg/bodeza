# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 17:05:07 2014

@author: Ross
"""



import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import cPickle as pickle


# Review json data
json_in = 'rotd_model_data.json'

# percentage of reviews to use all together
use_pct = 1

# percentage of used reviews to hold out for testing
test_pct = 0.1

# sgdc loss function to use
sgdc_loss = 'log' #'hinge

# load json data
with open(json_in) as fd:
     json_data = json.loads(fd.readline())

# calculate exact number of reviews to use and train
use_cutoff = int(round(len(json_data)*use_pct))
train_cutoff = int(round(use_cutoff*(1-test_pct)))

# organize train and test data
train = {'text':[], 'class':[]}
test = {'text':[], 'class':[]}
for i in range(use_cutoff):
    if i < train_cutoff:
        train['text'].append(json_data[i]['text'])
        train['class'].append(json_data[i]['type'])
    else:
        test['text'].append(json_data[i]['text'])
        test['class'].append(json_data[i]['type'])

# Set up classification work flow
sgdc_pipe = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss=sgdc_loss,penalty='l2',alpha=1e-5,n_iter=50))])

# Train the classifier
train_pipe = sgdc_pipe.fit_transform(train['text'], train['class'])

# If testing data is called for
if use_cutoff != train_cutoff:
    test_predict = sgdc_pipe.predict(test['text'])
    print(metrics.classification_report(test['class'], test_predict))
    test_predict_proba = sgdc_pipe.predict_proba(test['text'])
    print(metrics.log_loss(test['class'], test_predict_proba))    

# save the sgdc classifier
pickle.dump(sgdc_pipe, open("sgdc_pipe.p", "wb"))