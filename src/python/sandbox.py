# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 19:33:59 2014

@author: Ross
"""

import os
import json

os.chdir('C:/Users/Ross/Documents/GitHub/bodeza/yelp_data')

fd = open('yelp_academic_dataset_checkin.json')

data_list = []

for line in fd:
    line_data = line.strip()
    if line_data:
        data_list.append(json.loads(line))

fd.close()


temp = {'a':1}

temp['a']