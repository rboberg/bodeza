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
#import regex
import re
#need socket for timeouts
import socket

############# CONSTANTS
urlBase = "http://www.yelp.com/browse/reviews/picks_cities"
csvName = 'cities_mixed.csv'
noPages = 'all' #enter number of pages of reviews you want, per city. There are 10 reviews per page.
wait_min = 10 #in seconds. Might stop Yelp blocking you
wait_max = 20 

############# DEF FUNCTIONS
def getURLs(url,output):
    '''
    this function parses all the ROTD cities and writes the name,
    URL and # of pages to file for later access.
    '''
    #get page    
    soup = brewSoup(url)
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
    print '----------\n' + str(noPages) + ' pages of reviews in this city' + '\n-----------'   
    return noPages
    
def parseReviews(cityURL, noPages = 'all'):
    '''
    given a city URL to check out, this will parse the reviews up to the page
    limit given (reviews are on multiple pages). Default: parse every review.
    Writes to a text file for now. #NEEDS IMPROVEMENT
    '''
    #get page
    soup = brewSoup(cityURL)    
    #get the number of pages of reviews we need to traverse  
    if noPages == 'all':
        noPages = getNoPages(soup)
    #construct urls      
    urlList = []
    for i in range(noPages):
        #create url based on ROTD archive url rules.
        urlList.append(cityURL + '&start=' + str(i*10))
    #Randomize to stop detection
    random.shuffle(urlList)
    #traverse each page and save reviews
    reviewsList = []
    for i, url in enumerate(urlList):
        #get page
        soup = brewSoup(url)
        #save reviews
        reviews = soup.find_all('p',attrs={'lang':'en'})
        for review in reviews:
            #Review rating
            stars = review.parent.find('i', attrs={'title':re.compile('.star rating')})['title'].split(' ')[0]
            #Review date and ROTD designation date
            reviewDate = str(review.parent.find('span', attrs={'class':'rating-qualifier'}).text).strip()
            try:        
                ROTDDate = str(review.parent.find('a', attrs={'title':'Review of the Day'}).text).strip().split(' ')[1]       
            except:
                ROTDDate = 'NA'
            #Reviewer name, friends and review count
            numberFriends = str(review.parent.parent.parent.find('li', attrs={"class":"friend-count"}).text).split(' ')[1]       
            numberReviews = str(review.parent.parent.parent.find('li', attrs={"class":"review-count"}).text).split(' ')[1]       
            #Useful/Funny/Cool
            funny = str(review.parent.parent.find('a',attrs={'class':'ybtn ybtn-small funny'}).find('span', attrs={'class':'count'}).text)
            useful = str(review.parent.parent.find('a',attrs={'class':'ybtn ybtn-small useful'}).find('span', attrs={'class':'count'}).text)
            cool = str(review.parent.parent.find('a',attrs={'class':'ybtn ybtn-small cool'}).find('span', attrs={'class':'count'}).text)
            #Append data
            reviewsList.append([review.text,[stars,reviewDate,ROTDDate,numberFriends,numberReviews,funny,useful,cool]])
        #Print progress
        print 'Parsed page %i of %i... "%s"' %(i+1,noPages,url)
    #return result
    print '...All reviews parsed'
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
                f.write('<review# ' + str(i+1) + ',isUSA=' + str(isUSA) + ',' + ','.join(review[1]) + '>' + str(review[0]) + '\n')
    except:
        name = slugify(name)
        with open(os.path.join('results', name+'.txt'), 'wb') as f:
            for i, review in enumerate(reviewsList):
                try:
                    f.write('<review# ' + str(i+1) + ',isUSA=' + str(isUSA) + ',' + ','.join(review[1]) + '>' + str(review[0]) + '\n')
                except:
                    review[0] = slugify(review[0])
                    print name + 'review# ' + str(i) + ' was sanitised'
                    f.write('<review# ' + str(i+1) + ',isUSA=' + str(isUSA) + ',' + ','.join(review[1]) + '>' + str(review[0]) + '\n')
    print 'Wrote city %s to file' %cityName

##################### HELPER FUNCTIONS
def brewSoup(url):
    '''
    Load a html page, with simulation of web browser behaviors in case of crawler-stopper
    '''
    def waiter():
        '''
        waits for a time to prevent getting banned
        '''
        wait = wait_min + (wait_max - wait_min)*random.random()
        print 'waiting for %f seconds' %wait
        time.sleep(wait)    
    
    # different user agents to use to prevent getting banned.
    user_agents = [
        'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
        'Opera/9.25 (Windows NT 5.1; U; en)',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
        'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
        'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.142 Safari/535.19',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:11.0) Gecko/20100101 Firefox/11.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:8.0.1) Gecko/20100101 Firefox/8.0.1',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.151 Safari/535.19'
    ]  
    
    attempt = 0 #connection attempt #
    while True:
        attempt += 1
        try:    
            # chill out for a while
            waiter()  
              
            #choose a random user agent and download webpage       
            version = random.choice(user_agents)
            request = urllib2.Request(url)
            request.add_header('User-agent', version)
            page = urllib2.urlopen(request)
            soup = BeautifulSoup(page)
            return soup
        except socket.timeout as e:
            if attempt == 8:
                print 'Connection timed out on:',url,'\n',repr(e)
                raise e
        except Exception as e:
            print url,repr(e)
            raise e


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    value = unicode(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    value = unicode(re.sub('[-\s]+', '-', value))
    return value

##################### MAIN
#DEBUG
#getURLs(urlBase, csvName)

with open(csvName, 'r') as f:
    L = [line.strip().split(',') for line in f.readlines()]

for city in L:
    cityURL = '/'.join(urlBase.split('/')[:-3]) + city[0]
    print '----------\n' + 'About to parse reviews for: ' + str(city[1])  + '\n-----------'     
    reviews = parseReviews(cityURL, noPages)
    writeReviews(reviews,city[1],city[2])

"""
#############PARSE REVIEWS DEBUG
#get page
soup = brewSoup(cityURL)    
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
    #get page
    soup = brewSoup(url)
    #save reviews
    reviews = soup.find_all('p',attrs={'lang':'en'})
    for review in reviews:
        #Review rating
        stars = review.parent.find('i', attrs={'title':re.compile('.star rating')})['title'].split(' ')[0]
        #Review date and ROTD designation date
        reviewDate = str(review.parent.find('span', attrs={'class':'rating-qualifier'}).text).strip()
        try:        
            ROTDDate = str(review.parent.find('a', attrs={'title':'Review of the Day'}).text).strip().split(' ')[1]       
        except:
            ROTDDate = 'NA'
        #Reviewer name, friends and review count
        numberFriends = str(review.parent.parent.parent.find('li', attrs={"class":"friend-count"}).text).split(' ')[1]       
        numberReviews = str(review.parent.parent.parent.find('li', attrs={"class":"review-count"}).text).split(' ')[1]       
        #Useful/Funny/Cool
        funny = str(review.parent.parent.find('a',attrs={'class':'ybtn ybtn-small funny'}).find('span', attrs={'class':'count'}).text)
        useful = str(review.parent.parent.find('a',attrs={'class':'ybtn ybtn-small useful'}).find('span', attrs={'class':'count'}).text)
        cool = str(review.parent.parent.find('a',attrs={'class':'ybtn ybtn-small cool'}).find('span', attrs={'class':'count'}).text)
        #Append data
        reviewsList.append([review.text,[stars,reviewDate,ROTDDate,numberFriends,numberReviews,funny,useful,cool]])
    #Print progress
    print 'Parsed page %i of %i...' %(i+1,noPages)
#return result
print '...All reviews parsed'
return reviewsList   
############## / PARSE REVIEWS DEBUG
"""






#DEBUG:
'''
#save reviews
reviews = soup.find_all('p',attrs={'lang':'en'})
print reviews 
print 'and now length:'
print len(reviews)
for review in reviews:
    #DEBUG
    print "i'm in here"
    reviewsList.append(review.text)
'''

#of interest:
#http://sacharya.com/crawling-anonymously-with-tor-in-python/





