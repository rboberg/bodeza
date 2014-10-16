# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 19:33:59 2014

@author: Ross
"""
import os

import json

repo_dir = 'C:/Users/Ross/Documents/GitHub/bodeza'
data_dir = os.path.dirname(repo_dir) + '/bodeza_data'

fd = open(data_dir + '/yelp_data/yelp_academic_dataset_review.json')

data_list = []

for line in fd:
    line_data = line.strip()
    if line_data:
        data_list.append(json.loads(line))

fd.close()

