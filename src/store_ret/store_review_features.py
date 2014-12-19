# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:37:38 2014

This file assumes a sibling directory 'bodeza_data' that contains the
Yelp Academic Challenge Dataset.

Writes review data with added features from user and business collections.

@author: zszaiss
"""

import pymongo
import json

client = pymongo.MongoClient("mongodb://zsz:--------@ds048537.mongolab.com:48537/zsz-mongolab")
db = client["zsz-mongolab"]

# Only first 1000 reviews as a proof of concept.
import_limit = 1000

rev_collection = db["yelp_reviews"]
user_collection = db["yelp_users"]
biz_collection = db["yelp_businesses"]

reviews = open('bodeza_data/yelp_academic_dataset_review.json')

i = 0

for review in reviews:
    review_data = json.loads(review.strip())
    if review_data:
        review_data["_id"] = review_data[“review_id"]

        # Add additional features to store with review:
        #review_author = user_collection.find_one({"_id": "7_XwjOebd1temr3CaqGwpg"})
        review_author = user_collection.find_one({"_id": str(review_data["user_id"])})
        review_data["author_num_friends"] = len(review_author["friends"])
        review_data["author_num_fans"] = int(review_author["fans"])
        review_data["author_is_elite"] = (2014 in review_author["elite"])
        
        #review_biz = biz_collection.find_one({"_id": "nI1reikhvzQKyXojeUCPqg"})
        review_biz = biz_collection.find_one({"_id": str(review_data["business_id"])})
        review_data["biz_categories"] = review_biz["categories"]
        review_data["biz_avg_rating"] = review_biz["stars"]
        
        rev_collection.insert(review_data)
        i += 1
        if i >= import_limit: break

client.close()