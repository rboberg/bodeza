# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 14:42:07 2014

@author: Ross
"""
import sys
import re
import os
import json
from random import sample

def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    if len(argv) < 3:
        print "takes at least 3 arguments: self, rotd_folder, review_json_file"
        print "and optional arguments import_limit, out_fn, and mr_out"
        sys.exit(0)
    else:
        #rotd_folder = "C:/Users/Ross/Documents/GitHub/bodeza/beautiful soup/results"
        rotd_folder = argv[1]
        #review_json_file = 'C:/Users/Ross/Documents/GitHub/bodeza_data/yelp_data/yelp_academic_dataset_review.json'
        review_json_file = argv[2]
        
        import_limit = int(argv[3]) if len(argv) >= 4 else 2000000
        print import_limit        
        
        out_fn = argv[4] if len(argv) >= 5 else 'rotd_model_data.json'
        
        mr_out = argv[5] if len(argv) >= 6 else 'mrjob.txt'
            
        # Start list of ROTD Reviews
        rotd_reviews = []
        
        # Loop through files in ROTD folder
        i = 0
        for fn in os.listdir(rotd_folder):
            # If the file exists    
            if os.path.isfile(rotd_folder + '/' + fn):
                # Open the file
                fd = open(rotd_folder + "/" + fn)
                
                # Loop through ecah line in the file
                for line in fd:
                    # Extract the review and append it to the list
                    #rotd_reviews.append(''.join(re.split('<.*>',line.decode('ascii','ignore').encode('utf8','ignore'))).strip())
                    rotd_reviews.append(''.join(re.split('<.*>',line.decode('ascii','ignore').encode('ascii','ignore'))).strip())
                    i += 1
                    print str(i) + '/' + str(import_limit)
                    if i >= import_limit:
                        break
                
                # Close the file
                fd.close()
            
            if i >= import_limit:
                break

        print 'loaded ROTD'
        
        # load JSON data
        
        fd = open(review_json_file)
        
        base_data = []
        base_reviews = []
        i = 0
        for line in fd:
            line_data = line.strip()
            if line_data:
                base_data.append(json.loads(line))
                #base_reviews.append(base_data[i]['text'].encode('utf8'))
                base_reviews.append(base_data[i]['text'].encode('ascii', 'ignore'))
                i += 1
                print str(i) + '/' + str(import_limit)
                if i >= import_limit:
                    break
        
        
        fd.close()
        
        print 'loaded other reviews'
        
        # Create ROTD / NOT ROTD lists
        all_class = (['ROTD']*len(rotd_reviews)) + (['NotROTD']*len(base_reviews))
        all_review = rotd_reviews + base_reviews
        
        # Shuffle Lists
        strip_t_n = re.compile(r'[\t\n]')
        shuffle_order = sample([si for si in range(len(all_class))], len(all_class))
        rotd_model_data = []
        rotd_model_mrjob = []
        j = 0
        for i in shuffle_order:
            j += 1
            print j    
            review_text = all_review[i]#.decode('utf8','ignore').encode('ascii','ignore')
            rotd_model_data.append({'id':i, 'text':review_text, 'type':all_class[i]})
            rotd_model_mrjob.append(str(i) + "\t" + strip_t_n.sub(r' ',review_text))
        
        with open(out_fn, 'w') as out_fd:
            json.dump(rotd_model_data, out_fd)
        
        
        with open(mr_out, 'w') as out_fd:
            out_fd.write(u"\n".join(rotd_model_mrjob))
            
    
    

        
if __name__ == "__main__":
    sys.exit(main())

