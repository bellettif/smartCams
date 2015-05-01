'''
Created on May 1, 2015

    Generate background patches at random

@author: Francois Belletti
'''

from rect import Rect

N_TRIALS = 10

def generate_background(pict, bbxes, min_height, min_width):
    for i in xrange(N_TRIALS):
        rand_bbx = Rect(pict.shape[0], pict.shape[1])
        intersects = True
        if rand_bbx.height < min_height or rand_bbx.width < min_width:
            continue
        intersects = False
        for bbx in bbxes:
            intersects = intersects or rand_bbx.intersects(bbx)
        if not intersects:
            break
    if not intersects: 
        return pict[rand_bbx.ymin:rand_bbx.ymax,rand_bbx.xmin:rand_bbx.xmax]
    else:
        return None
    