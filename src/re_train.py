
import os
import negative
import train

def start():

	positive = '../DataBase/Positive'
	pos_gist = os.path.join(positive, str('Gist'))
	gist_list = os.listdir(pos_gist)
	hog_list = os.listdir(os.path.join(positive, str('Hog')))
	for x in gist_list:
		class_name = (x.split('.'))[0]
		negative.start(class_name, 20)
		print 'NEGATIVE ADDED FOR CLASS,', class_name
	print 'NEGATIVE ADDITION DONE'
	for x in gist_list:
		class_name = (x.split('.'))[0]
		train.start_1(class_name)
		print 'CLASSIFIER UPDATED FOR CLASS,', class_name
	print 'CLASSIFIER UPDATED'
