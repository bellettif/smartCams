'''
Created on May 1, 2015

@author: cusgadmin
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import caffe
import cPickle as pickle
from sklearn import *

from hybrid import Hybrid_classifier

HYBRID = False

lookup_table    = '../image_dump/label_lookup_table.pi'
image_folder            = '../image_dump/cropped/'
rcnn_feature_folder     = '../VOC2012/rcnn_features/' 
fc7_feature_folder      = '../VOC2012/fc7_features/'

#
#    Setup filepaths to models (NN and maybe SVM or other)
#
if HYBRID:
    model_file      = 'rcnn_model/deploy.prototxt'
    model_weights   = 'rcnn_model/bvlc_reference_rcnn_ilsvrc13.caffemodel'
    head_path       = '../rcnn_features_ml/svm_fc7_no_bg_model'
else:
    model_file      = '../finetuning/rcc_net/deploy_nn.prototxt'
    model_weights   = '../finetuning/rcc_net/no_background/caffenet_train_iter_7000.caffemodel'

if HYBRID:
    classifier = Hybrid_classifier(model_file,
                                   model_weights,
                                   head_path,
                                   label_lookup = lookup_table)
else:
    classifier = Hybrid_classifier(model_file,
                                   model_weights,
                                   label_lookup = lookup_table)
    
image_list = filter(lambda x : x[-4:] == '.jpg', os.listdir(image_folder))

example_image = caffe.io.load_image(image_folder + 'pict_1025.jpg')

num_label = classifier.classify(example_image, target_feature = 'fc7')

print num_label

plt.title(num_label)
plt.imshow(example_image)
plt.show()


