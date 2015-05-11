'''
Created on May 1, 2015

@author: cusgadmin
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle
from sklearn import *
import caffe

from hybrid import Hybrid_classifier

HYBRID = True

lookup_table            = '../image_dump/label_lookup_table.pi'
image_folder            = '../image_dump/cropped/'
rcnn_feature_folder     = '../VOC2012/rcnn_features/' 
fc7_feature_folder      = '../VOC2012/fc7_features/'

#
#    Setup filepaths to models (NN and maybe SVM or other)
#
if HYBRID:
    model_file      = 'rcnn_model/deploy.prototxt'
    model_weights   = '../finetuning/bvlc_reference_rcnn_ilsvrc13.caffemodel'
    head_path       = '../rcnn_features_ml/svm_fc7_with_bg_model'
else:
    model_file      = '../finetuning/rcc_net/deploy_nn_background.prototxt'
    model_weights   = '../finetuning/rcc_net/background/caffenet_train_background_iter_10000.caffemodel'

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

for i in xrange(1000):
    example_image = caffe.io.load_image(image_folder + np.random.choice(image_list))
    num_label = classifier.classify(example_image, target_feature = 'fc7')
    plt.title(classifier.lookup(num_label))
    plt.imshow(example_image)
    plt.show()