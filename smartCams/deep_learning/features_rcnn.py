'''
Created on Apr 30, 2015

@author: Francois Belletti
'''

import numpy as np
import caffe
import cv2
import os
import cPickle as pickle

MODEL_FILE = 'rcnn_model/deploy.prototxt'
PRETRAINED = 'rcnn_model/bvlc_reference_rcnn_ilsvrc13.caffemodel'

image_folder    = '../finetuning/warped_data/'
feature_folder  = '../VOC2012/rcnn_features/' 

image_list = filter(lambda x : x[-4:] == '.jpg', os.listdir(image_folder))

#
#    Set up caffe classifier
#
caffe.set_mode_cpu()
#
#    No need for mean image here, normalization has already been done
#
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(227, 227))

#
#    Iterate through all the normalized images and compute rcnn features
#
for image_file in image_list:
    print image_file
    file_path = image_folder + image_file
    input_image = cv2.imread(file_path)
    #
    features = net.predict([input_image])
    print features.shape
    #
    #pickle.dump(features, open(feature_folder + image_file[:-4] + '.pi', 'wb'))
