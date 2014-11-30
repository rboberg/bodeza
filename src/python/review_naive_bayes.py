# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 19:49:04 2014

@author: Ross
"""
# Modules
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify import MaxentClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import  BigramAssocMeasures
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from math import log
import os
import json
import pandas as pd
import numpy as np


# Tokenize
def my_tokenize(text, ftokenize = nltk.word_tokenize, stemmer = None, stopset = set(), bigrams = False):
    w_list = ftokenize(text)
    if stemmer is None:   
        w_list = [w.lower() for w in w_list if not w.lower() in stopset]
    else:
        w_list = [stemmer.stem(w.lower()) for w in w_list if not w.lower() in stopset]
    
    if bigrams:
        b_list = []
        for i in range(len(w_list)-1):
            b_list.append(w_list[i] + w_list[i+1])
        w_list.extend(b_list)
            
    return w_list


# load JSON data
repo_dir = 'C:/Users/Ross/Documents/GitHub/bodeza'
data_dir = os.path.dirname(repo_dir) + '/bodeza_data'
fd = open(data_dir + '/yelp_data/yelp_academic_dataset_review.json')

import_limit = 50000

data_list = []
i = 0
for line in fd:
    line_data = line.strip()
    if line_data:
        data_list.append(json.loads(line))
        i += 1
        if i >= import_limit: break


fd.close()

train_size = 25000
test_size = 25000

# Data manipulation function
def get_tokens(data_i, stemmer=None):
    ftokenize = RegexpTokenizer(r"\w+").tokenize
    #ftokenize = nltk.word_tokenize
    stopset = set(stopwords.words('english'))
    #stopset = set()
    
    # Removing punctuation and including stopset
    # reduced number of tokens by 37% and
    # significantly increased r-squared
    
    tokens = my_tokenize(data_i['text'], ftokenize = ftokenize, stemmer=stemmer, stopset=stopset)
    return tokens    
    #feat = {w : True for w in tokens}
    #return feat

stemmer = PorterStemmer()

# tokenize all data
all_tokens = []
for i in range(len(data_list)):
    tokens = get_tokens(data_list[i], stemmer=stemmer)
    data_list[i]['tokens'] = tokens
    all_tokens.extend(tokens)


# Get training data


train_feats = []
for data_i in data_list[:train_size]:
    train_feats.append(({w : True for w in data_i['tokens']}, data_i['stars']))    


# Get testing data


test_feats = []
test_stars = []
for data_i in data_list[train_size:(train_size+test_size)]:
    test_feats.append({w : True for w in data_i['tokens']})   
    test_stars.append(data_i['stars'])

######################
# NAIVE BAYES

# Train the data
nbc = NaiveBayesClassifier.train(train_feats)
nbc.most_informative_features(10)

# Get classification (from max probability)
model_stars = nbc.classify_many(test_feats)

# Get star as weighted average probability
model_star_prob = nbc.prob_classify_many(test_feats)

def prob_weight(prob_dist):
    return sum([prob_dist.prob(star)*star for star in prob_dist.samples()])
model_weighted = [prob_weight(prob_dist) for prob_dist in model_star_prob]

# Test the model
star_df = pd.DataFrame.from_dict({
    'actual':test_stars,
    'model_max_prob':model_stars,
    'model_weighted':model_weighted,
    'c':2.5
    })
pd.ols(y=star_df['actual'], x=star_df['model_max_prob'])

pd.ols(y=star_df['actual'], x=star_df['model_weighted'])

# trained on 10k
# r^2 = 11.5%

# trained on 50k
# r^2 = 13%

# trained on 100k
# r^2 = 13.7%



#############################
# DECISION TREE CLASSIFIER
# note: this doesn't seem to work to well for our data

# Trains slower than naive bayes
# entropy_cutoff is used to create new branches
### entropy is the uncertainty of the outcome
### 1 = high uncertainty
### 0 = low uncertainty

# depth_cutoff controls the depth of the tree
### the default value is 100
# support_cuttoff
### controls how many labeled feature sets are required to refine the tree

# Train the data
dtc = DecisionTreeClassifier.train(train_feats, entropy_cutoff = 0.8, depth_cutoff = 5, support_cutoff=30, verbose = True)

# Get classification (from max probability)
dtc_stars = dtc.classify_many(test_feats)

# Test the model
star_df['model_dtc'] = dtc_stars

pd.ols(y=star_df['actual'], x=star_df['model_dtc'])

pd.ols(y=star_df['model_max_prob'], x=star_df['model_dtc'])

pd.ols(y=star_df['actual'], x=star_df[['model_dtc','model_weighted']])

pd.ols(y=star_df['actual'], x=(star_df['model_dtc'] + star_df['model_weighted'])/2)



###############################
# Max Entropy Classifier
# note: this also doesn't seem to work well for our data

# gis is faster than default iis
# can interupt with ctrl c and still get a valid model
mec = MaxentClassifier.train(train_feats, trace = 3, max_iter=10, min_lldelta=0.1, algorithm = 'gis')

# Get classification (from max probability)
mec_stars = mec.classify_many(test_feats)

# Test the model
star_df['model_mec'] = mec_stars


pd.ols(y=star_df['actual'], x=star_df['model_mec'])

##########################
##########################
# Try a Bigram Only Model
# note: not more sucessful

bigram_train = []
for data_i in data_list[:train_size]:
    tokens = data_i['tokens']
    bigram_train.append(({(tokens[i], tokens[i+1]) : True for i in range(len(tokens)-1)}, data_i['stars']))    

bigram_test = []
for data_i in data_list[train_size:(train_size+test_size)]:
    tokens = data_i['tokens']
    bigram_test.append({(tokens[i], tokens[i+1]) : True for i in range(len(tokens)-1)})   



# Train the data
nbc_bigram = NaiveBayesClassifier.train(bigram_train)
nbc_bigram.most_informative_features(10)

# Get classification (from max probability)
model_bigram_stars = nbc_bigram.classify_many(bigram_test)

# Get star as weighted average probability
model_bigram_star_prob = nbc_bigram.prob_classify_many(bigram_test)

def prob_weight(prob_dist):
    return sum([prob_dist.prob(star)*star for star in prob_dist.samples()])
model_bigram_weighted = [prob_weight(prob_dist) for prob_dist in model_bigram_star_prob]

# Test the model
star_df['model_bigram_max'] = model_bigram_stars
star_df['model_bigram_weighted'] = model_bigram_weighted

pd.ols(y=star_df['actual'], x=star_df['model_bigram_weighted'])

pd.ols(y=star_df['model_bigram_weighted'], x=star_df['model_weighted'])


########
### Try combining word and bigram model probabilities
### note: not more succesful
def entropy(probs):
    return -1*sum([p*log(p) for p in probs])


combo_prob = []
combo_weighted = []
combo_max = []
for i in range(len(model_star_prob)):
    
    dist1 = model_star_prob[i]
    dist2 = model_bigram_star_prob[i]
    stars = dist1.samples()
    prob1 = [dist1.prob(star) for star in stars]
    prob2 = [dist2.prob(star) for star in stars]
    entr1 = entropy(prob1)
    entr2 = entropy(prob2)
    p1 = entr2 / (entr1 + entr2)
    
    combo_prob.append({star: dist1.prob(star)*p1 + dist2.prob(star)*(1-p1) for star in dist1.samples()})
    combo_weighted.append(sum([k * v for (k, v) in combo_prob[i].iteritems()]))
    combo_max.append(dist1.max() if p1 < 0.5 else dist2.max())


star_df['model_combo_weighted'] = combo_weighted
star_df['model_combo_max'] = combo_max

pd.ols(y=star_df['actual'], x=star_df['model_combo_weighted'])
pd.ols(y=star_df['actual'], x=star_df['model_combo_max'])



###############################
### sklearn package test

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

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
for i in range(train_size):
    train_docs.append(data_list[i]['text'])
    star_train = np.append(star_train,data_list[i]['stars'])

# Create term document matrix
count_v = CountVectorizer(tokenizer=ftokenize, stop_words='english')

td_train = count_v.fit_transform(train_docs)
count_v.get_feature_names()

# Create term document times inverse document frequency matrix
tfidf_transform = TfidfTransformer()
tfidf_train = tfidf_transform.fit_transform(td_train)


# Train the model
mnbc = MultinomialNB().fit(tfidf_train, star_train)

# Get testing data
test_docs = []
star_test = np.array([])
for i in range(train_size,train_size+test_size):
    test_docs.append(data_list[i]['text'])
    star_test = np.append(star_test,data_list[i]['stars'])

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

### Support Vector Machine (SVM) 
#svmc = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-10, n_iter=50)
svmc = SGDClassifier(loss='log', penalty='l2', alpha=1e-10, n_iter=50)
svmc.fit(tfidf_train, star_train)

svmc_pred = svmc.predict(tdidf_test)
svmc_pred

# Acccuracy
np.mean(star_test == svmc_pred)

### More advanced accuracy tests

svmc.predict_proba(tdidf_test)

print(metrics.classification_report(star_test, mnbc_pred))
print(metrics.classification_report(star_test, svmc_pred))



print(metrics.classification_report(star_test, np.array(star_df['model_max_prob'])))
print(metrics.classification_report(star_test, np.array([1 for i in range(len(star_test))])))

print(metrics.confusion_matrix(star_test, mnbc_pred))
print(metrics.confusion_matrix(star_test, svmc_pred))
mnbc_conf = metrics.confusion_matrix(star_test, mnbc_pred)
svmc_conf = metrics.confusion_matrix(star_test, svmc_pred)

(svmc_conf.astype(float) / np.apply_along_axis(sum, 0, svmc_conf)).round(2)


print(metrics.confusion_matrix(star_test, star_df['model_max_prob']))

print(metrics.confusion_matrix(star_test, np.array([1 for i in range(len(star_test))])))

star_df['model_weighted'].round()

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





