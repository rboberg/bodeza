# -*- coding: utf-8 -*-
"""
Created on Sun Nov 2 17:15:09 2014

This file assumes a sibling directory 'bodeza_data' that contains the
Yelp Academic Challenge Dataset.

@author: zszaiss
"""

import pymongo
import json

client = pymongo.MongoClient("mongodb://zsz:--------@ds048537.mongolab.com:48537/zsz-mongolab")
db = client["zsz-mongolab"]

biz_collection = db["yelp_businesses"]
businesses = open('bodeza_data/yelp_academic_dataset_business.json')

import_limit = 500000
i = 0

for biz in businesses:
    biz_data = json.loads(biz.strip())
    if biz_data:
        biz_data["_id"] = biz_data["business_id"]
        biz_collection.insert(biz_data)
        i += 1
        if i >= import_limit: break


user_collection = db["yelp_users"]
users = open('bodeza_data/yelp_academic_dataset_user.json')

i = 0

for user in users:
    user_data = json.loads(user.strip())
    if user_data:
        user_data["_id"] = user_data["user_id"]
        user_collection.insert(user_data)
        i += 1
        if i >= import_limit: break

client.close()