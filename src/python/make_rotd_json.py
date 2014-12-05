# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 16:45:21 2014

@author: Ross
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 14:42:07 2014

@author: Ross
"""

import re
import os
import json
from random import sample

# ROTD text file folder
rotd_folder = "C:/Users/Ross/Documents/GitHub/bodeza/beautiful soup/results"
review_json_file = 'C:/Users/Ross/Documents/GitHub/bodeza_data/yelp_data/yelp_academic_dataset_review.json'
out_fn = 'rotd_model_data.json'
import_limit = 2000000

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

fd = open(review_json_file)

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

# Shuffle Lists
shuffle_order = sample([si for si in range(len(all_class))], len(all_class))
rotd_model_data = []
for i in shuffle_order:
    rotd_model_data.append({'text':all_review[i], 'type':all_class[i]})

with open(out_fn, 'w') as out_fd:
    json.dump(rotd_model_data, out_fd)

