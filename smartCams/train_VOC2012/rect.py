'''
Created on May 1, 2015

    Class that encapsulates bounding boxes

@author: Francois Belletti
'''

import numpy as np

class Rect:
    
    #
    #    Initializes random with the size the input image
    #
    def __init__(self, height, width):
        #
        self.ymin = int(np.random.uniform(low = height * 0.1, high = height * 0.9))
        self.ymax = int(np.random.uniform(low = self.ymin, high = height))
        self.xmin = int(np.random.uniform(low = width * 0.1, high = width * 0.9))
        self.xmax = int(np.random.uniform(low = self.xmin, high = width))
        #
        self.height = self.ymax - self.ymin
        self.width  = self.xmax - self.xmin
        #
        
    #
    #    Returns true if self intersects with bbx
    #
    def intersects(self, bbx):
        b_width     = bbx['xmax'] - bbx['xmin']
        b_height    = bbx['ymax'] - bbx['ymin']
        return (abs(self.xmin - bbx['xmin']) * 2 < (self.width + b_width)) and \
                    (abs(self.ymin - bbx['ymin']) * 2 < (self.height + b_height))