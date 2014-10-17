# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:49:04 2014

@author: Ross
"""
# Modules
import nltk
from nltk.classify import NaiveBayesClassifier
import os
import json
import pandas as pd

# Tokenize
def my_tokenize(text):
    w_list = nltk.word_tokenize(data_i['text'])
    w_list = [w.lower() for w in w_list]
    return w_list

repo_dir = 'C:/Users/Ross/Documents/GitHub/bodeza'
data_dir = os.path.dirname(repo_dir) + '/bodeza_data'

fd = open(data_dir + '/yelp_data/yelp_academic_dataset_review.json')

data_list = []

for line in fd:
    line_data = line.strip()
    if line_data:
        data_list.append(json.loads(line))

fd.close()

train_size = 10000

train_feats = []
for data_i in data_list[:train_size]:
    star = data_i['stars']    
    feat = {w : True for w in my_tokenize(data_i['text'])}
    train_feats.append((feat, star))

test_size = 10000

test_feats = []
test_stars = []
for data_i in data_list[train_size:(train_size+test_size)]:
    test_stars.append(data_i['stars'])
    feat = {w : True for w in my_tokenize(data_i['text'])}
    test_feats.append(feat)

nbc = NaiveBayesClassifier.train(train_feats)
model_stars = nbc.classify_many(test_feats)

star_df = pd.DataFrame.from_dict({'actual':test_stars,'model':model_stars, 'c':2.5})

pd.ols(y=star_df['actual'], x=star_df['model'])

# trained on 10k
# r^2 = 11.5%

# trained on 50k
# r^2 = 13%

# trained on 100k
# r^2 = 13.7%