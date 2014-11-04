# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 17:36:25 2014

@author: Miggles
"""
############# IMPORT MODULES
from bs4 import BeautifulSoup
#urllib is to query the actual web page
import urllib2
#csv to write csv
import csv
#time to timestamp files
import time
#os to write results in a subdirectory
import os
#random wait times to help prevent yelp from blocking you :(
import random

############# CONSTANTS
urlBase = "http://www.yelp.com/browse/reviews/picks_cities"
csvName = 'cities.csv'
noPages = 1 #enter number of pages of reviews you want, per city. There are 10 reviews per page.
wait_min = 10 #in seconds. Might stop Yelp blocking you
wait_max = 50 

############# DEF FUNCTIONS
def getURLs(url,output):
    '''
    this function parses all the ROTD cities and writes the name,
    URL and # of pages to file for later access.
    '''
    #Query the website and save the html
    page = urllib2.urlopen(url)    
    #parse into a beautiful soup object
    soup = BeautifulSoup(page)
    #find the URLs to different cities -
    #I cheated and looked at the source code of the webpage to figure out
    #the structure
    lvl2node = soup.find_all('div',attrs={'class':"column-alpha"})
    cities = []
    isInUSA = True
    count = 0
    for lvl3node in lvl2node[0]:
        #One of the children is the heading indicating international cities
        if 'International' in str(lvl3node):
            isInUSA = False
        #Some children are 'states' which in turn have the links as children
        if 'class="states"' in str(lvl3node):
            count +=1
            #iterate through cities
            for city in lvl3node:
                if 'a href' in str(city):
                    #Save url, city name, and 'is in USA' boolean
                    cities.append((city.a.get('href'),city.a.string.encode('UTF-8'),isInUSA))   
    #write to csv file                
    with open(output, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(cities)

def getNoPages(soup):
    '''
    Finds how many pages of reviews there are given a city.
    returns an integer.
    called by getReviews()
    '''
    s = soup.find('div',attrs={'class':"page-of-pages"})
    #extract the total number of pages from formatted text
    noPages = int(s.string.split('\n')[-2].split(' ')[-1])
    print '----------\n\n\n' + str(noPages) + '\n\n\n\n-----------'   
    return noPages

def waiter():
    wait = wait_min + (wait_max - wait_min)*random.random()
    print 'waiting for %f seconds' %wait
    time.sleep(wait)
    
def parseReviews(cityURL, noPages = 'all'):
    '''
    given a city URL to check out, this will parse the reviews up to the page
    limit given (reviews are on multiple pages). Default: parse every review.
    Writes to a text file for now. #NEEDS IMPROVEMENT
    '''
    #Chill out for a while
    waiter()    
    #Query the base website and save the html
    page = urllib2.urlopen(cityURL)    
    #parse into a beautiful soup object
    soup = BeautifulSoup(page)    
    #get the number of pages of reviews we need to traverse  
    if noPages == 'all':
        noPages = getNoPages(soup)
    #construct urls      
    urlList = []
    for i in range(noPages):
        #create url based on ROTD archive url rules
        urlList.append(cityURL + '&start=' + str(i*10))
    #traverse each page and save reviews
    reviewsList = []
    for i, url in enumerate(urlList):
        #Query the website and save the html
        page = urllib2.urlopen(url)        
        #parse into a beautiful soup object
        soup = BeautifulSoup(page)        
        #save reviews
        reviews = soup.find_all('p',attrs={'class':'review_comment ieSucks'})
        for review in reviews:
            reviewsList.append(review.text)
                #Print progress
        print 'Parsed page %i of %i' %(i+1,noPages)
    #return result
    print 'all reviews parsed'
    return reviewsList   

def writeReviews(reviewsList,cityName,isUSA):
    '''
    writes reviews to file.
    needs overhauling!
    '''
    #Naming convention    
    t = int(time.time()) #accurate closest second
    name = str(t) + '_' + cityName
    #Make subdirectory  
    try:
        os.mkdir('results')
    except OSError:
        pass #folder already exists
    
    try:
        with open(os.path.join('results', name+'.txt'), 'wb') as f:
            for i, review in enumerate(reviewsList):
                f.write('<review# ' + str(i+1) + ',' + 'isUSA=' + str(isUSA) + '>' + review + '\n')
    except:
        name = slugify(name)
        with open(os.path.join('results', name+'.txt'), 'wb') as f:
            for i, review in enumerate(reviewsList):
                try:
                    f.write('<review# ' + str(i+1) + ',' + 'isUSA=' + str(isUSA) + '>' + review + '\n')
                except:
                    review = slugify(review)
                    print name + 'review# ' + str(i) + ' was sanitised'
                    f.write('<review# ' + str(i+1) + ',' + 'isUSA=' + str(isUSA) + '>' + review + '\n')
    print 'Wrote city %s to file' %cityName

#def removeNonAscii(s): 
#    return "".join(i for i in s if ord(i)<128)

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    import re
    value = unicode(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    value = unicode(re.sub('[-\s]+', '-', value))
    return value

##################### MAIN
getURLs(urlBase, csvName)

with open(csvName, 'r') as f:
    L = [line.strip().split(',') for line in f.readlines()]

for city in L:
    cityURL = '/'.join(urlBase.split('/')[:-3]) + city[0]
    reviews = parseReviews(cityURL, noPages)
    writeReviews(reviews,city[1],city[2])
