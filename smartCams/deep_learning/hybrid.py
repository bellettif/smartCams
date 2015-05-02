'''
Created on May 1, 2015

    Hybrid NN for featurization / SVM or other for classification
    The topmost classifier is called the ``head''.

@author: Francois Belletti
'''

import numpy as np
import sklearn as sk
import caffe
import cPickle as pickle

caffe.set_mode_cpu()

class Hybrid_classifier:
    
    #
    #    Instantiates caffe model and head classifier
    #    @param model_filepath     String path to caffe model prototxt
    #    @param weight_filepath    String path to model's weights
    #    @param head_filepath      String path to shallow featurizer
    #    @param label_lookup       String path to label lookup table
    #    @param mean_path          String path to mean image
    def __init__(self, model_filepath, 
                 weight_filepath,
                 head_filepath, 
                 label_lookup,
                 mean_path = 'ilsvrc_2012_mean.npy'):
        self.net            =   caffe.Net(model_filepath, weight_filepath, caffe.TEST)
        self.head           =   pickle.load(open(head_filepath, 'rb'))
        self.label_lookup   =   pickle.load(open(label_lookup, 'rb'))
        self.mean_image     =   np.load(mean_path)
        self.transformer    =   caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', np.load('ilsvrc_2012_mean.npy').mean(1).mean(1))
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_channel_swap('data', (2,1,0))
        self.transformer.set_raw_scale('data', 255.0)
    
    #
    #    Featurize a given image, works with a file path or an image
    #
    def featurize(self, input_image, target_layers = ['fc7']):
        if type(input_image) is str:
            im = caffe.io.load_image(input_image)
            out = self.net.forward_all(target_layers, 
                                       data = np.asarray([self.transformer.preprocess('data', im)]))
        else:
            out = self.net.forward_all(target_layers, 
                                       data = np.asarray([self.transformer.preprocess('data', input_image)]))
        return [out[x] for x in target_layers]
    
    #
    #    Classify a given image, works with a file path or an image
    #
    def classify(self, input_image, target_feature, probas = False):
        target_layers = [target_feature]
        if type(input_image) is str:
            im  = caffe.io.load_image(input_image)
            out = self.featurize(im, target_layers)
        else:
            out = self.featurize(im, target_layers)
        feature_vect = out[0]
        if not probas:
            return self.head.predict(feature_vect)
        else:
            return self.head.predict_log_proba(feature_vect)