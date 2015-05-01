'''
Created on Apr 30, 2015

    Proceed with feature extraction with the rcnn network

@author: Francois Belletti
'''

import numpy as np
import caffe
import os
import cPickle as pickle
from matplotlib import pyplot as plt

MODEL_FILE = 'rcnn_model/deploy.prototxt'
PRETRAINED = 'rcnn_model/bvlc_reference_rcnn_ilsvrc13.caffemodel'

image_folder    = '../image_dump/cropped/'
rcnn_feature_folder  = '../VOC2012/rcnn_features/' 
fc7_feature_folder  = '../VOC2012/fc7_features/'

image_list = filter(lambda x : x[-4:] == '.jpg', os.listdir(image_folder))

#
#    Set up caffe classifier
#
caffe.set_mode_cpu()
#
#    No need for mean image here, normalization has already been done
#
net = caffe.Net('rcnn_model_surgery/deploy.prototxt',
                'rcnn_model_surgery/bvlc_reference_rcnn_ilsvrc13.caffemodel',
                caffe.TEST)

def get_features(image_path):
    input_image = caffe.io.load_image(image_path)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    # make classification map by forward and print prediction indices at each location
    out = net.forward_all(['fc-rcnn', 'fc7'], data=np.asarray([transformer.preprocess('data', input_image)]))
    # Return 200 and 4096 features
    return out['fc-rcnn'], out['fc7']

existing_feats_files = 0
for i, image_name in enumerate(image_list):
    rcnn_feats_filename = rcnn_feature_folder + image_name[:-4] + '.pi'
    fc7_feats_filename = fc7_feature_folder + image_name[:-4] + '.pi'
    #
    rcnn_feats, fc7_feats = get_features(image_folder + image_name)    
    pickle.dump(rcnn_feats, open(rcnn_feats_filename, 'wb'))
    pickle.dump(fc7_feats, open(fc7_feats_filename, 'wb'))
    if i % 100 == 0:
        print 'Featurized image %d out of %d' % (i, len(image_list))

print existing_feats_files, 'feature files already existed.'
