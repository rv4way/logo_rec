import numpy as np
import cPickle
import os
import cv2
import pandas as pd

def calc_hist(img_arr):

	rows, cols = img_arr.shape[:2]
	basehsv = cv2.cvtColor(img_arr,cv2.COLOR_BGR2HSV)

	hbins = 180
	sbins = 255
	hrange = [0,180]
	srange = [0,256]
	ranges = hrange+srange

	img_hist = cv2.calcHist(basehsv,[0,1],None,[180,256],ranges)
	cv2.normalize(img_hist,img_hist,0,255,cv2.NORM_MINMAX)

	return img_hist

def save_hist(img_hist, profile_id):
	file_name = '../DataBase/histogram/' + str(profile_id) + '.pkl'

	f = open(file_name, 'wb')
	cPickle.dump(img_hist, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

def search_hist(profile_id, img_arr):
	hist_path = '../DataBase/histogram/'
	img_hist = calc_hist(img_arr)
	response = {}
	for x in profile_id:
		file_path = os.path.join(hist_path, x)
		file_path = file_path + '.pkl'
		data = open(file_path, 'rb')
		hist = cPickle.load(data)

		distance = cv2.compareHist(img_hist,hist,0)
		print distance, file_path
		response[x] = distance
	print response


def add_hist(img_arr, profile_id):
	img_hist = calc_hist(img_arr)
	save_hist(img_hist, profile_id)