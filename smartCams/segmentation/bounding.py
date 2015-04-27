'''
Created on Apr 23, 2015

    Bounding boxes and ellipses

@author: Francois
'''

import cPickle as pickle
import cv2
from matplotlib import pyplot as plt
import numpy as np

img = pickle.load(open('../seg_example.pi', 'rb'))

img += np.abs(np.min(img))

img = 255.0 * img / float(np.max(img) - np.min(img))

img_png = np.zeros(img.shape, dtype = np.uint8)

img_png[:,:] = img[:,:]

img_png = cv2.blur(img_png, (16, 16))

thr = 90

ret, thresh3 = cv2.threshold(img_png, thr, 255, cv2.THRESH_TOZERO_INV)
ret, thresh4 = cv2.threshold(img_png, 255 - thr, 255, cv2.THRESH_TOZERO)

thresh = np.zeros(img.shape, dtype = np.uint8)

thresh[:] = ((thresh3 + thresh4) * 0.5 )[:]

contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

sub_picts = []
entropies = []
for contour in contours:
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect
    print rect
    if (w < 2) or (h < 2):
        continue
    sub_pict = img[y:(y + h), x:(x + w)]
    sub_picts.append(sub_pict)
    #
    probas, _ = np.histogram(np.ravel(sub_pict), bins = 256)
    probas = 1.0 * probas[np.where(probas > 0)]
    probas /= float(np.sum(probas))
    entropy = - np.sum(probas * np.log(probas))
    #
    #plt.subplot(211)
    #plt.imshow(sub_pict)
    #plt.subplot(212)
    #plt.title('Entropy = %f' % entropy)
    #plt.hist(np.ravel(sub_pict), bins = 256)
    #plt.show()
    cv2.rectangle(img_png, (x, y), (x + w, y + h), (255, 255, 255))
    
plt.imshow(img_png, cmap = plt.get_cmap('gray'))
plt.show()