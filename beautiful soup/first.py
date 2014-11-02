# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:36:25 2014

@author: Miggles
"""

from bs4 import BeautifulSoup

#urllib is to query the actual web page
import urllib2



urlBase = "http://www.yelp.com/browse/reviews/picks?start="
reviewsList = []
no_pages = 317 #WILL HAVE TO COME UP WITH A BETTER WAY TO DO THIS
    
for i in range(no_pages):
    #create url based on ROTD archive url rules
    url = urlBase + str(i*10) 
    #Query the website and save the html
    page = urllib2.urlopen(url)
    
    #parse into a beautiful soup object
    soup = BeautifulSoup(page)
    
    #save reviews
    reviews = soup.find_all('p',attrs={'class':'review_comment ieSucks'})
    for review in reviews:
        reviewsList.append(str(review))
    
    for review in reviewsList:
        print '\n\n\n ----------------------------------------------- \n\n\n'
     #   if len(review) > 1:
     #       print 'offending review:'
     #       print review        
     #       raise ValueError('review variable should be a list of only one element, right? right??')
        print review

#average review is ~1729 chars and 
#min(words)
#18
#>>> max(words)
#1340
#average(words): 283
