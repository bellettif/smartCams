'''
Created on Apr 27, 2015

    Extract cropped and warped images from VOC
    Dump all files in a given folder
    The folder will contain the list of images and labels in a txt file.
    The folder will contain a list of images for training and a list for testing.
    The folder will also contain the jpg mean image.
    A subfolder will contain all images with their mean_substracted the other one the original images.

@author: Francois Belletti
'''

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import cPickle as pickle

from back_ground import generate_background

#
#
#
TEST_PROBA      = 0.1  # Put that proportion of images into the test set, the rest into the learning set
IMAGE_WIDTH     = 227  # Width of the warped images
IMAGE_HEIGHT    = 227  # Height of the warped images
CAFFE_ROOT      = '/users/cusgadmin/caffe'
TARGET_LABELS   = ['background', 'person', 'bicycle', 'bus', 'car', 'motorbike'] # These elements only will be extracted
#
#    Setup path to images and annotations (annotations contain labels and bounding boxes in xml format)
#
MAIN_DIR            = '/Users/cusgadmin/smartCams/Wksp/smartCams/'
INPUT_IMAGES        = MAIN_DIR + 'VOC2012/JPEGImages/'
ANNOTATIONS         = MAIN_DIR + 'VOC2012/Annotations/'
BACK_GROUND_JPG     = MAIN_DIR + 'background/images/'
BACK_GROUND_AN      = MAIN_DIR + 'background/annotation/n04335209/'
OUTPUT_FOLDER       = '../image_dump/'
N_BACK_PER_IMAGE    = 100
MIN_BACK_HEIGHT     = 40
MIN_BACK_WIDTH      = 40
#

#
#    Convert string labels to integers, pickle corresponding table
#
num_labels = dict(zip(TARGET_LABELS, range(len(TARGET_LABELS))))
pickle.dump(num_labels, open(OUTPUT_FOLDER + 'label_lookup_table.pi', 'wb'))

#
#    Extract bounding box coordinates from the corresponding xml annotation file
#
def extract_bbxes(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = root.findall('object')
    #
    bbxes = []
    size_xml = root.find('size')
    size = {'width' : size_xml.find('width').text,
            'height' : size_xml.find('height').text,
            'depth' : size_xml.find('depth').text}
    for x in objects:
        bbx = x.find('bndbox')
        bbxes.append({
                      'name' : x.find('name').text,
                      'xmin' : int(float(bbx.find('xmin').text)),
                      'ymin' : int(float(bbx.find('ymin').text)),
                      'xmax' : int(float(bbx.find('xmax').text)),
                      'ymax' : int(float(bbx.find('ymax').text))
                      })
    return size, bbxes

#
#    Filter out the files that do not have the extension
#
def filter_file_list(file_list, extension = '.xml'):
    return filter(lambda x : x[-4:] == extension, file_list)

#
#    Prepare input and output
#
annotation_list = filter_file_list(os.listdir(ANNOTATIONS))
file_list_train         =   open(OUTPUT_FOLDER + 'train_list.txt', 'wb')
file_list_test          =   open(OUTPUT_FOLDER + 'test_list.txt', 'wb')
file_list_train_test    =   open(OUTPUT_FOLDER + 'train_test_list.txt', 'wb')
#
#    Prepare folder structure
#
if 'cropped' not in os.listdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER + 'cropped')
#
#    First pass on the dataset without mean substraction
#
print 'Passing on data set of objects'
#
n_selected = 0

for i, annotation in enumerate(annotation_list):
    file_path = ANNOTATIONS + annotation
    size, bbxes = extract_bbxes(file_path)
    bbxes = filter(lambda x : x['name'] in TARGET_LABELS, bbxes)
    if len(bbxes) == 0:
        continue
    #    Load image to crop it
    image_path = INPUT_IMAGES + annotation[:-4] + '.jpg'
    pict = cv2.imread(image_path)
    #
    if i % 100 == 0:
        print 'Proccessing annotation %d out of %d' % (i, len(annotation_list))
    #
    for bbx in bbxes:
        #
        if 'cropped' not in os.listdir(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER + 'cropped')
        #
        output_path = OUTPUT_FOLDER + ('cropped/pict_%d.jpg' % n_selected)
        #    Write out image file path and labels
        if np.random.uniform() > (1.0 - TEST_PROBA):
            file_list_test.write(output_path + ' ' + str(num_labels[bbx['name']]) + '\n')
        else:
            file_list_train.write(output_path + ' ' + str(num_labels[bbx['name']]) + '\n')
        file_list_train_test.write(output_path + ' ' + str(num_labels[bbx['name']]) + '\n')
        #    Compute cropped image
        sub_image = pict[bbx['ymin']:bbx['ymax'], bbx['xmin']:bbx['xmax']].copy()   # Crop
        sub_image = cv2.resize(sub_image, (IMAGE_HEIGHT, IMAGE_WIDTH))              # Warp
        #
        cv2.imwrite(output_path, sub_image)
        #
        n_selected += 1

print n_selected
print 'Passing on data set of background picts'
#
annotation_list = filter_file_list(os.listdir(BACK_GROUND_AN))
#
print len(annotation_list)
for i, annotation in enumerate(annotation_list):
    file_path = BACK_GROUND_AN + annotation
    size, bbxes = extract_bbxes(file_path)
    if len(bbxes) == 0:
        continue
    #    Load image to crop it
    image_path = BACK_GROUND_JPG + annotation[:-4] + '.JPEG'
    #
    pict = cv2.imread(image_path)
    #
    if i % 100 == 0:
        print 'Proccessing annotation %d out of %d' % (i, len(annotation_list))
    #
    for ii in range(N_BACK_PER_IMAGE):
        # Try and extract background
        back_ground = generate_background(pict, bbxes, MIN_BACK_HEIGHT, MIN_BACK_WIDTH)
        if back_ground is not None:
            output_path = OUTPUT_FOLDER + ('cropped/pict_%d.jpg' % n_selected)
            #    Put filepath in list of files
            if np.random.uniform() > (1.0 - TEST_PROBA):
                file_list_test.write(output_path + ' ' + str(num_labels['background']) + '\n')
            else:
                file_list_train.write(output_path + ' ' + str(num_labels['background']) + '\n')
            file_list_train_test.write(output_path + ' ' + str(num_labels['background']) + '\n')
            #    Write background image to disk
            #
            back_ground = cv2.resize(back_ground, (IMAGE_HEIGHT, IMAGE_WIDTH))
            cv2.imwrite(output_path, back_ground)
            #
            n_selected +=1
            #
        else:   # No background has been found
            #print 'No background found'
            continue
print n_selected
#
#    Closing output
#
file_list_train.close()
file_list_test.close()
file_list_train_test.close()