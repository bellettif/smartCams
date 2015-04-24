'''
Created on Apr 22, 2015

@author: Francois Belletti
'''

import cPickle as pickle
from matplotlib import pyplot as plt
import os

all_images = pickle.load(open('all_images.pi', 'rb'))

print all_images