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

in_file_path = '../../../Data/My_videos/angle_1.MOV'
out_file_path = '../cam_imgs/'

videocapt = cv2.VideoCapture(in_file_path)

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
    
def extract_frame(fr):
    videocapt.set(cv.CV_CAP_PROP_POS_FRAMES, int(fr));
    _, temp = videocapt.read()
    return cv2.resize(temp, (0, 0), fx = rescale_factor, fy = rescale_factor)

blur_radius = 6

for i, fr in enumerate(extract_idx):
    all_images[i] = cv2.blur(extract_frame(fr), (blur_radius, blur_radius))

median_image = np.median(all_images, axis = 0)
low_image    = np.percentile(all_images, 25, axis = 0)
high_image   = np.percentile(all_images, 75, axis = 0)
std_image    = np.std(all_images, axis = 0)

cv2.imwrite(out_file_path + 'median_image.jpg', median_image)
cv2.imwrite(out_file_path + 'high_image.jpg',   high_image)
cv2.imwrite(out_file_path + 'low_image.jpg',    low_image)
cv2.imwrite(out_file_path + 'std_image.jpg',    std_image)

begin_sequence  = int(0.8 * frame_count)
end_sequence    = int(0.85 * frame_count)
n_extraced      = 40

extract_idx = np.linspace(begin_sequence, end_sequence, n_extraced)

for i, fr in enumerate(extract_idx):
    cv2.imwrite(out_file_path + str(i) + '.jpg', extract_frame(fr))

videocapt.release()