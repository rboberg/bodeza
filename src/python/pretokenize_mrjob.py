# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:02:49 2014

The make_rotd_json file above also creates a text file for mrjob processing. The mapper code can be run on that text file as:

	$ python pretokenize_mrjob.py mrjob.txt | processed.txt

This returns key value pairs of review id's and space separated strings of tokens that can be further processed by scikit-learn classifiers.

@author: Ross
"""
from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def ftokenize(text, stemmer=PorterStemmer(), tokenizer=RegexpTokenizer(r"\w+").tokenize):
    stops = stopwords.words('english')
    tokens = tokenizer(text)
    return [stemmer.stem(w.lower()) for w in tokens if w.lower() not in stops ]

class MRTokenize(MRJob):
    OUTPUT_PROTOCOL = JSONProtocol
    
    def mapper(self, _, value):
        # separate on tabs        
        tab_sep = value.split("\t");
        # tokenize and rejoin
        tokenized = " ".join(ftokenize(tab_sep[1]))
        yield tab_sep[0], tokenized
        
if __name__ == '__main__':
    MRTokenize.run();