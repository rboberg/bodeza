# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 17:05:07 2014

make_rotd_json.py combines ROTD and Yelp Academic Data Set Review Data in to a single JSON.
It takes 5 arguments:
    1. path to folder holding ROTD scrape results for each city
    2. path to JSON file for academic dataset review
    3. max number of reviews to load for each data set
    4. combined JSON data file output
    5. name of text file for mrjob


@author: Ross
"""



import json
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import cPickle as pickle


def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    if len(argv) < 4:
        print 'takes at least 4 arg:'
        print '* self'
        print '* the json in file'
        print '* percent of reviews to use all together as decimal'
        print '* percent of used reviews to hold out for testing as decimal, 0 if none'
        print 'One optional parameter:'
        print '* the filename of the classifier pipeline to pickle, if you want to save it'
    else:
        #json_in = 'rotd_model_data.json'
        json_in = argv[1]
        
        use_pct = float(argv[2])
        test_pct = float(argv[3])
        
        #classifier_pickle = "sgdc_pipe.p"        
        classifier_pickle = argv[4] if len(argv) >= 5 else None
        
        # sgdc loss function to use
        sgdc_loss = 'log'
        
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
            print 'Classification Report:'            
            print(metrics.classification_report(test['class'], test_predict))
            test_predict_proba = sgdc_pipe.predict_proba(test['text'])
            print 'Log Loss:'            
            print(metrics.log_loss(test['class'], test_predict_proba))    
        
        if not classifier_pickle is None:
            # save the sgdc classifier
            pickle.dump(sgdc_pipe, open(classifier_pickle, "wb"))
            

if __name__ == "__main__":
    sys.exit(main())

