# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 08:09:46 2014

predict_rotd.py takes raw review text and the pickled classifier and prints the probability that the review came from the ROTD data set, not the academic data set.

    $ python predict_rotd.py "review text here" sgdc_pipe.p
    0.0678734814354

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
        # Get arguments
        review_text = argv[1]
        pipe_pickle = argv[2]
        
        # Load pickled classifier
        classifier_pipe = pickle.load(open(pipe_pickle,"rb"))
        
        # Get prediction probabilities        
        probs = classifier_pipe.predict_proba([review_text])[0]
        print probs[1]
        
if __name__ == "__main__":
    sys.exit(main())

