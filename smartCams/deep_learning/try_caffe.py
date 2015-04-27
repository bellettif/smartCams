'''
Created on Apr 26, 2015

@author: Francois Belletti
'''

import caffe

import sys
import caffe
import cv2
import Image
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import numpy as np
import lmdb
caffe_root = 'home/francois/'

MODEL_FILE = '%s/examples/mnist/lenet.prototxt' % caffe_root
PRETRAINED = '%s/examples/mnist/lenet_iter_10000.caffemodel' % caffe_root

net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
caffe.set_mode_cpu()
db_path = '%s/examples/mnist/mnist_test_lmdb'
lmdb_env = lmdb.open(db_path)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
count = 0
correct = 0
for key, value in lmdb_cursor:
    print "Count:"
    print count
    count = count + 1
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum)
    image = image.astype(np.uint8)
    out = net.forward_all(data=np.asarray([image]))
    predicted_label = out['prob'][0].argmax(axis=0)
    if label == predicted_label[0][0]:
        correct = correct + 1
    print("Label is class " + str(label) + ", predicted class is " + str(predicted_label[0][0]))
    if count == 3:
        break
print(str(correct) + " out of " + str(count) + " were classified correctly")
