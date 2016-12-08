import Gist_feat_last
import HOG_feat2
import os
import numpy as np
from sklearn.externals import joblib  #save the data
import cv2
import json
from pymongo import MongoClient
import afine_search
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
    print final_gist
    return final_gist
    

                
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
    print final_rv
    return final_rv

def gen_res(final_rv, final_gist):
     #print 'KINGKINGKINGKING', final_gist.values()
     #print 'ncjdsndjnkdnfkd', final_rv.values()
     temp_gist = final_gist.keys()
     temp_hog = final_rv.keys()
     #print temp_gist
     #print temp_hog
     #print final_gist
     #print final_rv
     final_temp = list(set(temp_gist).intersection(temp_hog))
     #print final_temp
     gist_count = []
     hog_count = []
     response = []
     for x in final_temp:
        gist_count.append(final_gist[x])
        hog_count.append(final_rv[x])
     m_gist = max(gist_count)
     m_hog = max(hog_count)
     #print gist_count
     #print hog_count
     #print m_gist
     #print m_hog
     if m_hog > m_gist:
          for x in final_rv:
              com = final_rv[x]
          if com == m_hog:
               response.append(x)
     elif m_gist > m_hog:
          for x in final_gist:
            com = final_gist[x]
            if com == m_gist:
               response.append(x)
     elif m_gist == m_hog:
          return final_temp
     #print 'KJJHGGHJFHGDFDJGHKIULIKJGFHG',response
     return response
           
         
    
def image_calc(img):
    #cv2.imshow('Original', img)
    #cv2.waitKey()
    try:
        correct_fea = Gist_feat_last.singleImage2(img)
        feat = HOG_feat2.hog_call(img)
        #print 'hog',feat.shape
        #print 'correct shape',correct_fea.shape
        final_rv = Label_classify2(feat,'batman')
        final_gist = Label_classify(correct_fea,'batman')
        #print 'list',list1
        #print '2nd list',list2
        orig_res = gen_res(final_rv, final_gist)

        af_img = afine_search.affine_transform(img)

        af_gist = Gist_feat_last.singleImage2(af_img)
        af_hog = HOG_feat2.hog_call(af_img)
        final_rv = Label_classify2(feat,'batman')
        final_gist = Label_classify(correct_fea,'batman')
        af_res = gen_res(final_rv, final_gist)

        cv2.imshow('Affine', af_img)
        cv2.waitKey()

        final_res = list(set(orig_res).intersection(af_res))
        #print final_res
        return final_res

    except Exception,e:
        return 'Image not found'

if __name__ == '__main__':
    path = '/home/rahul/Downloads/3m_test.png'
    img = cv2.imread(path)
    image_calc(img)