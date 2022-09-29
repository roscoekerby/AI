# imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import pandas as pd
from IPython.display import display
from multiprocessing import Process

# show images without blocking using multiprocessing
def plot():
	plt.title(query_file_name + ' => ' + file_name)
	plt.get_current_fig_manager().set_window_title(query_file_name + ' => ' + file_name)	
	plt.imshow(draw_matches,), plt.show()   

# data for dataframe
data = {
  	"query_file_name": [],
  	"matched_file_name": [],
  	"match_points": []
	}

# initialise counter for number of files and set total number of query files	
file_count = 1
number_of_query_files = len(next(os.walk('query/'))[2])

# iterate through each file in folder query to look for matches
# please make sure you have a query folder in the same directory as this code
for query_file_name in os.listdir("query/"):
	best_file_name = None # initialise best file 
	best_number_matches = 0 # initialise best number of matches
	query_image = cv.imread('query/' + query_file_name, cv.IMREAD_GRAYSCALE) # current query file
	
	# iterate through each file in folder query_against to look for matches to query folder
	# please make sure you have a query_against folder in the same directory as this code
	for file_name in os.listdir("query_against/"):
		current_image = cv.imread("query_against/" + file_name, cv.IMREAD_GRAYSCALE) # current comparing file
		# Initiate SIFT detector
		sift = cv.SIFT_create()
		# find the keypoints (kp) and descriptors (des) with SIFT
		kp1, des1 = sift.detectAndCompute(query_image, None)
		kp2, des2 = sift.detectAndCompute(current_image, None)
		# FLANN parameters
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)   # or pass empty dictionary
		flann = cv.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, des2, k = 2)
		# store all the good matches as per Lowe's ratio test.
		match_points = []
		# Need to draw only good matches, so create a mask
		matches_mask = [[0, 0] for i in range(len(matches))]
		match_count = 0 # initialise number of matches to 0 for each new comparing file
		
		# ratio test as per Lowe's paper
		for i, (m, n) in enumerate(matches):
		    if m.distance < 0.7 * n.distance:
		    	# point has been found
		    	matches_mask[i]=[1, 0]
		    	match_count = match_count + 1
		    	match_points.append(m)
		
		draw_params = dict(matchColor = (0, 255, 0),
				   singlePointColor = (255, 0, 0),
				   matchesMask = matches_mask,
				   flags = cv.DrawMatchesFlags_DEFAULT)
					  			      
		if match_count > best_number_matches: # new highest number of matches		
			best_number_matches = match_count # number of match points
			best_file_name = file_name # closest file matching query image so far
			draw_matches = cv.drawMatchesKnn(query_image, kp1, current_image, kp2, matches, None, **draw_params) # draws matching lines
	
	# add data to data dictionary
	data["query_file_name"].append(query_file_name)
	data["matched_file_name"].append(best_file_name)
	data["match_points"].append(best_number_matches)

	if (best_number_matches > 100): # show good images without blocking using multiprocessing
		p = Process(target = plot)
		p.start()
		
	print("{}/{}".format(file_count, number_of_query_files)) # indicates the progess of the program 
	file_count = file_count + 1 # next file 

#load data into a DataFrame object:
df = pd.DataFrame(data)

def force_show_all(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        display(df)

# display df:
force_show_all(df)





