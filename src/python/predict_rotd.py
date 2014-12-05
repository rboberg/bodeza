# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 08:09:46 2014

@author: Ross
"""

import sys
import cPickle as pickle

def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    if len(argv) < 3:
        print "takes at least 3 arguments: self, review, classifier pickle"
        sys.exit(0)
    else:
        pipe_pickle = argv[2]
        review_text = argv[1]
        classifier_pipe = pickle.load(open(pipe_pickle,"rb"))
        return classifier_pipe.predict_proba([review_text])[0]
        
if __name__ == "__main__":
    sys.exit(main())

