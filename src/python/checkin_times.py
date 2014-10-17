# -*- coding: utf-8 -*-
"""
Created on Thu Oct 09 19:33:59 2014

@author: Ross
"""

import os
import json
import pandas as pd
from ggplot import *

repo_dir = 'C:/Users/Ross/Documents/GitHub/bodeza'
data_dir = os.path.dirname(repo_dir) + '/bodeza_data'

fd = open(data_dir + '/yelp_data/yelp_academic_dataset_checkin.json')

data_list = []

for line in fd:
    line_data = line.strip()
    if line_data:
        data_list.append(json.loads(line))

fd.close()


checkins = {}

for checkin in data_list:
    for time, n in checkin['checkin_info'].iteritems():
        if time in checkins:
            checkins[time] += 1
        else:
            checkins[time] = 1

freq_dict = [dict(zip(['h','d','n'], ki.split('-')+[vi])) for ki, vi in checkins.iteritems()]

freq_df = pd.DataFrame(freq_dict)

print ggplot(freq_df, aes(x='h', y='n', colour='d')) + geom_line()

