# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 14:42:07 2014

@author: Ross
"""

import re
import os
import json
from random import sample
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
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


# ROTD text file folder
rotd_folder = "C:/Users/Ross/Documents/GitHub/bodeza/beautiful soup/results"

# Start list of ROTD Reviews
rotd_reviews = []

# Loop through files in ROTD folder
for fn in os.listdir(rotd_folder):
    # If the file exists    
    if os.path.isfile(rotd_folder + '/' + fn):
        # Open the file
        fd = open(rotd_folder + "/" + fn)
        
        # Loop through ecah line in the file
        for line in fd:
            # Extract the review and append it to the list
            rotd_reviews.append(''.join(re.split('<.*>',line.encode('utf8'))).strip())
        
        # Close the file
        fd.close()

# load JSON data
json_folder = 'C:/Users/Ross/Documents/GitHub/bodeza_data'
fd = open(json_folder + '/yelp_data/yelp_academic_dataset_review.json')

import_limit = 150000

base_data = []
base_reviews = []
i = 0
for line in fd:
    line_data = line.strip()
    if line_data:
        base_data.append(json.loads(line))
        base_reviews.append(base_data[i]['text'])
        i += 1
        if i >= import_limit: break


fd.close()

# Create ROTD / NOT ROTD lists
all_class = (['ROTD']*len(rotd_reviews)) + (['NotROTD']*len(base_reviews))
all_review = rotd_reviews + base_reviews

# Set training / testing amounts
train_on_pct = 0.9
train_i = range(int(round(train_on_pct * len(all_class))))
test_i = range(len(train_i), len(all_class))


# Shuffle Lists & Create Train / Test Samples
shuffle_order = sample([si for si in range(len(all_class))], len(all_class))
train_class = []
train_review = []
for i in [shuffle_order[tri] for tri in train_i]:
    train_class.append(all_class[i])
    train_review.append(all_review[i])
    
test_class = []
test_review = []
for i in [shuffle_order[tei] for tei in test_i]:
    test_class.append(all_class[i])
    test_review.append(all_review[i])




# Create tokenizer function
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r"\w+").tokenize
def ftokenize(text):
    tokens = tokenizer(text)
    return [stemmer.stem(w) for w in tokens]

# Create term document matrix
count_v = CountVectorizer(tokenizer=ftokenize, stop_words='english')
td_train = count_v.fit_transform(train_review)
td_test = count_v.transform(test_review)

# Create term document times inverse document frequency matrix
tfidf_transform = TfidfTransformer()
tfidf_train = tfidf_transform.fit_transform(td_train)
tfidf_test = tfidf_transform.transform(td_test)

# Stochastic Gradient Descent Classifier
sgdc_loss = 'log' #'hinge
sgdc = SGDClassifier(loss=sgdc_loss, penalty='l2', alpha=1e-5, n_iter=50)
sgdc.fit(tfidf_train, train_class)

sgdc_pred = sgdc.predict(tfidf_test)
sgdc_pred

print(metrics.classification_report(test_class, sgdc_pred))

# Check out arbitrary review
sgdc.predict_proba(tfidf_transform.transform(count_v.transform(["I hate this restaurant"])))


### Try to use a Pipeline

sgdc_pipe = Pipeline([('vect', CountVectorizer(tokenizer=ftokenize, stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss=sgdc_loss,penalty='l2',alpha=1e-5,n_iter=50))])

train_pipe = sgdc_pipe.fit_transform(train_review, train_class)

sgdc_pipe.predict_proba(['area bay pickle cured braised avocado wrap bacon doughnut lamb cheese menu Exquisite Everyday Appetizer Dessert Wine Owner Leftover Music Live Talk Date night'])
