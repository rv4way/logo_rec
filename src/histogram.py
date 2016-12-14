import numpy as np
import matplotlib.pyplot as plt
import cPickle
import os

def cal_hist(img_arr):
	
	img_arr = img_arr.mean(axis=2).flatten()
	img_b, img_bins, img_patches = plt.hist(img_arr, 255)

	return img_bins

def save_bins(img_bins, profile_id):
	file_name = '../database/histogram/' + str(profile_id) + '.pkl'
	f = open(file_name, 'wb')
	cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

def search_hist(profile_id, img_arr):
	hist_path = '../database/histogram/'
	img_bins = cal_hist(img_arr)
	response = {}
	for x in profile_id:
		file_path = os.path.join(hist_path, x)
		data = open(file_path, 'rb')
		bins = cPickle.load(data)
		distance = img_bins - bins
		distance = format(np.dot(distance, distance))
		response[x] = distance




def histo(img_arr, profile_id):
	img_bins = cal_hist(img_arr)
	save_bins(img_bins, profile_id)