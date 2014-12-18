bodeza
======

MIDS Fall 2014 Project

<table>
<tr><th colspan=2>authors</th></tr>
<tr><td>ross</td><td>BO berg</td></tr>
<tr><td>michael</td><td>DE mertzi</td></tr>
<tr><td>sam</td><td>ZA iss</td></tr>
</table>

## General

### Directory Structure

	bodeza/
		src/ (for all the source code)
			python/ (for python scripts)
		www/ (web files for final deliverable)
		beautiful_soup/ (scraper code and results)

### Yelp Data
The yelp data we're working on is too big for github. So the code for now assumes that the bodeza repo on your local machine has the same parent directory as another folder called "bodeza_data", which has a subfolder "yelp_data".

So the parent directory looks like:
	
	parent/
	  bodeza (this is the folder where your git repo is)
	  bodeza_data/
	    yelp_data/ (this is the folder with the JSON files)
  
## Creating the Classifier

Code stored in and commands run out of:
	
	bodeza/src/python

### Step1 Combine Data Sources

make_rotd_json.py combines ROTD and Yelp Academic Data Set Review Data in to a single JSON. It takes 5 arguments:
<ol>
	<li> path to folder holding ROTD scrape results for each city </li>
	<li> path to JSON file for academic dataset review </li>
	<li> max number of reviews to load for each data set </li>
	<li> combined JSON data file output </li>
	<li> name of text file for mrjob </li>
</ol>

Running for testing on a subset:
	
	$ python make_rotd_json.py ../../beautiful_soup/results ../../../bodeza_data/yelp_data/yelp_academic_dataset_review.json 20000 rotd_model_sub.json mrjob_sub.txt

Running for production:
	
	$ python make_rotd_json.py ../../beautiful_soup/results ../../../bodeza_data/yelp_data/yelp_academic_dataset_review.json 2000000 rotd_model.json mrjob.txt

### Step2 Train & Save

make_rotd_classifier.py trains the classifier and pickles a scikit-learn pipeline object that can be used to evaluate raw text reviews. It takes 4 arguments:
<ol>
	<li>combined JSON file output from step 1</li>
	<li>percent of the JSON data to use in building the model as a decimal - useful for running on subsets of the data for testing</li>
	<li>percent of the used data held out for testing as a decimal</li>
	<li>the name of the file to save the classifier pipeline pickle</li>
</ol>

Running for testing on a subset:
	
	$ python make_rotd_classifier.py rotd_model_data.json 0.1 0.1 sgdc_pipe_sub.p

Running for production holding out 10% for testing:
	
	$ python make_rotd_classifier.py rotd_model_data.json 1 0.1 sgdc_pipe.p
	$ python make_rotd_classifier.py rotd_model_data.json 1 0.1 sgdc_pipe.p
	Classification Report:
	             precision    recall  f1-score   support

	    NotROTD       0.92      0.99      0.95    112594
	       ROTD       0.79      0.23      0.36     13109

	avg / total       0.90      0.91      0.89    125703

	Log Loss:
	0.214131894687


Running for production with no testing:
	
	$ python make_rotd_classifier.py rotd_model_data.json 1 0 sgdc_pipe.p

### Step3 Use It

predict_rotd.py takes raw review text and the pickled classifier and prints the probability that the review came from the ROTD data set, not the academic data set.
	
	$ python predict_rotd.py "review text here" sgdc_pipe.p
	0.0678734814354
