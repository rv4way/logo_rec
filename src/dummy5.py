import Gist_feat_last
import HOG_feat2
import os
import numpy as np
from sklearn.externals import joblib  #save the data
import cv2
import json
from pymongo import MongoClient
import afine_search
import histogram
from scipy.misc import imresize
with open('Properties.json', 'r') as fp:
    data = json.load(fp)

dir_gist = data["ClassifierGist"]
dir_hog = data["ClassifierHog"]
m_a = data["MongoUrl"]

c = MongoClient(m_a)   #taking instance of mongo client
mer4 = data["ImageDatabase"]
db = c[mer4] 
db_classig = db.ClasiGabor
db_classih = db.ClassiHog
list1 = {} #dictionary to hold data
list2 = {}
list3 = {}
files_name = []

orgi_gist = {}
orig_hog = {}
aff_gist = {}
aff_hog = {}




#just return the name of company of classiffier
def remove_num(list_temp):
    for i in range(len(list_temp)):
        temp = list_temp[i]
        temp1 = temp.split('_')
        temp = temp1[0]
        list_temp[i] = temp
        
    return list_temp



def Label_classify(feature,files1):
    final_gist = {}
    dir2 = dir_gist #directory where the classifier are
    for subdir2,newdir1,files3 in os.walk(dir2):
        list1[files1]=[]
        files_name.append(files1)
        for files4 in files3:
            machine_path = dir2+'/'+files4
            profile_id = ((files4.split('_'))[0])
            clf = joblib.load(machine_path) #load the classifier
            predict = clf.predict(feature) #predict the class
            predict = np.asarray(predict)
            if predict.all()==1:  #if class is one then add it                
                list1[files1].append(files4)
                if final_gist.has_key(profile_id):
                     temp = final_gist[profile_id]
                     temp += 1
                     final_gist[profile_id] = temp
                else:
                     final_gist[profile_id] = int(1)
    #print final_gist
    return final_gist
    #return list1
    

                
def Label_classify2(feature,files1):
    final_rv = {}
    dir2 = dir_hog #directory where the classifier are
    for subdir2,newdir1,files3 in os.walk(dir2):
        list2[files1]=[]
    #print files3
        for files4 in files3:
            machine_path = dir2+'/'+files4
            profile_id = (files4.split('_'))[0]
            clf = joblib.load(machine_path) #load the classifier
            predict = clf.predict(feature) #predict the class
            predict = np.asarray(predict)
            if predict.all()==1:  #if class is one then add it
                #print 'Prediction is:',files4
                list2[files1].append(files4)
                if final_rv.has_key(profile_id):
                     temp = final_rv[profile_id]
                     temp += 1
                     final_rv[profile_id] = temp
                else:
                     final_rv[profile_id] = int(1)
    #print final_rv
    return final_rv
    #return list2

def gen_res(final_rv, final_gist):
    temp_gist = final_gist.keys()
    temp_hog = final_rv.keys()
    final_temp = list(set(temp_gist).intersection(temp_hog))
    #print temp_gist
    #print temp_hog
    #print final_temp
    return final_temp      
         
    
def image_calc(img):

    img = imresize(img, (47*2, 55*2), interp = 'bicubic')
    correct_fea = Gist_feat_last.singleImage2(img)
    feat = HOG_feat2.hog_call(img)

    orig_hog = Label_classify2(feat,'batman')
    orig_gist = Label_classify(correct_fea,'batman')

    orig_res = gen_res(orig_hog, orig_gist)

    af_img = afine_search.affine_transform(img)

    af_gist = Gist_feat_last.singleImage2(af_img)
    af_hog = HOG_feat2.hog_call(af_img)
    aff_hog = Label_classify2(feat,'batman')
    aff_gist = Label_classify(correct_fea,'batman')
    af_res = gen_res(aff_hog, aff_gist)

    final_res = list(set(orig_res).intersection(af_res))

    final_response = histogram.search_hist(final_res, img)
    return final_response

def search(img_arr):

    res1 = image_calc(img_arr)
    in_logo = afine_search.inside_logo(img)
    
    res2 = image_calc(in_logo)

    res3 = list(set(res1).intersection(res2))

    print res3

'''
if __name__ == '__main__':
    path = '/home/rahul/Dropbox/Processed Logo Images/Logos Phase 2/Budweiser/all_B/Correct/download_jpg_logo_300_56217b_1.png'
    img = cv2.imread(path)
    search(img)
'''
