# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:02:49 2014

@author: Ross
"""
from mrjob.job import MRJob
from mrjob.protocol import JSONProtocol
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#stemmer = PorterStemmer()
#tokenizer = RegexpTokenizer(r"\w+").tokenize
def ftokenize(text, stemmer=PorterStemmer(), tokenizer=RegexpTokenizer(r"\w+").tokenize):
    stops = stopwords.words('english')
    tokens = tokenizer(text)
    return [stemmer.stem(w.lower()) for w in tokens if w.lower() not in stops ]


class MRTokenize(MRJob):
    OUTPUT_PROTOCOL = JSONProtocol
    
    def mapper(self, _, value):
        tab_sep = value.split("\t");
        tokenized = " ".join(ftokenize(tab_sep[1]))
        yield tab_sep[0], tokenized
        
if __name__ == '__main__':
    MRTokenize.run();