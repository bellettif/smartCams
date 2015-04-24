'''
Created on Apr 22, 2015

    How to open a video stream

@author: Francois Belletti
'''

import cv2
import cv
from matplotlib import pyplot as plt
import numpy as np
import cPickle as pickle

file_path = '../../../Data/My_videos/angle_1.MOV'

print file_path

videocapt = cv2.VideoCapture(file_path)

_, example = videocapt.read()

print example.shape

frame_rate =  videocapt.get(cv.CV_CAP_PROP_FPS)
frame_count = int(videocapt.get(cv.CV_CAP_PROP_FRAME_COUNT))
frame_height = videocapt.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
frame_width = videocapt.get(cv.CV_CAP_PROP_FRAME_WIDTH)
n_channels = 3

print frame_rate
print frame_count
print frame_height
print frame_width

rescale_factor = 0.5

extract_idx = np.linspace(0, frame_count, 100)
n_extraced = len(extract_idx)

all_images = np.zeros((n_extraced, frame_height * 0.5, frame_width * 0.5, n_channels), dtype = np.uint8)
    
for i, fr in enumerate(extract_idx):
    videocapt.set(cv.CV_CAP_PROP_POS_FRAMES, int(fr));
    _, temp = videocapt.read()
    all_images[i] = cv2.resize(temp, (0, 0), fx = rescale_factor, fy = rescale_factor)

videocapt.release()

all_images = np.asanyarray(all_images)

print 'Pickling'
pickle.dump(all_images, open('all_images.pi', 'wb'))
print 'Pickling done'

grey_scale = np.mean(all_images, axis = -1)

plt.subplot(221)
plt.imshow(np.median(grey_scale, axis = 0), cmap = plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(np.mean(grey_scale, axis = 0), cmap = plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(np.std(grey_scale, axis = 0), cmap = plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(grey_scale[0] - np.median(grey_scale, axis = 0), cmap = plt.get_cmap('gray'))
plt.show()

