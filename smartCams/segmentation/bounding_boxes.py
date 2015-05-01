'''
Created on Apr 23, 2015

    Bounding boxes and ellipses from extracted video images

@author: Francois
'''

import cPickle as pickle
import cv2
from matplotlib import pyplot as plt
import numpy as np

#
#    Extract bounding boxes from pre-processed image
#
#    @param seg A gray scale image
#    @param thr The threshold value (90 to 110 is reasonnable) for contour detection
#    @param blur Blur radius in number of pixels
#
#    @return List of bounding box coordinates (cv2 rectangles)
#
def get_bbx(seg, thr = 100, blur = 6):
    #
    #    Normalized picture
    #
    norm_pct =  np.copy(seg)
    norm_pct += np.abs(np.min(norm_pct))
    norm_pct =  255.0 * norm_pct / float(np.max(norm_pct) - np.min(norm_pct))
    norm_pct =  norm_pct.astype(np.uint8)
    #
    #    Blur the picture
    #
    blurred_pct = cv2.blur(norm_pct, (blur, blur))
    #
    #    Extract thresholds
    #
    ret, thresh1 = cv2.threshold(blurred_pct, thr, 255, cv2.THRESH_TOZERO_INV)
    ret, thresh2 = cv2.threshold(blurred_pct, 255 - thr, 255, cv2.THRESH_TOZERO)
    thresh = np.zeros(norm_pct.shape, dtype = np.uint8)
    thresh[:] = ((thresh1 + thresh2) * 0.5 )[:]
    #
    #    Extract contours
    #
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    #    Return list of contours
    #
    return contours

#
#   Example
#
input_folder = '../processed_imgs/'
output_folder = '../region_proposals/'
frame = 47
#
dump = pickle.load(open('%sseg_example_%d.pi' % (input_folder, frame), 'rb'))
#
seg         = dump['seg']
original    = dump['original'].astype(np.uint8)
#
contours = get_bbx(seg)
#
sub_picts = []
for contour in contours:
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    print rect
    if (w < 2) or (h < 2):
        continue
    y_b = max(y - 0.5 * h, 0)
    y_e = min(y + 1.5 * h, original.shape[0])
    x_b = max(x - 0.5 * w, 0)
    x_e = min(x + 1.5 * w, original.shape[1])
    sub_pict = original[y_b:y_e, x_b:x_e]
    sub_picts.append(sub_pict)
    #cv2.rectangle(original, (x, y), (x + w, y + h), (255, 255, 255))

for i, sub_pict in enumerate(sub_picts):
    if sub_pict.shape[0] > 100 or sub_pict.shape[1] > 100:
        cv2.imwrite('../region_proposals/%d_%d_bbx.png' % (frame, i), sub_pict)