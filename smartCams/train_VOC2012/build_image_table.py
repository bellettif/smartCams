'''
Created on Apr 27, 2015

    Consolidate the VOC image set around cars, persons and bicycles
    Extract cropped and warped windows from the dataset so as to use the Pascal VOC network
    Builds the mean image of the data set

@author: Francois Belletti
'''

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
#import sys
#sys.path.insert(0, '/Users/alvinyuan/Documents/Vision/caffe/python')
import caffe.io

MAX_N_IMAGES = 1e3
TEST_PROBA   = 0.1  # Put that proportion of images into the test set, the rest into the learning set
IMAGE_WIDTH  = 227  # Width of the warped images
IMAGE_HEIGHT = 227  # Height of the warped images
CAFFE_ROOT   = '/Users/alvinyuan/Documents/Vision/caffe'

data_set_suffix = 'small'

main_dir            = '/Users/cusgadmin/smartCams/Wksp/smartCams/'
#main_dir            = '/Users/alvinyuan/Documents/Vision/smartCams/'
image_set_folder    = main_dir + 'VOC2012/ImageSets/Main/'
jpeg_folder         = main_dir + 'VOC2012/JPEGImages/'
annotation_folder   = main_dir + 'VOC2012/Annotations/'
output_cropped      = '../warped_data/'

all_sets = os.listdir(image_set_folder)

target_labels = ['person', 'bicycle', 'bus', 'car', 'motorbike']
num_labels = {'person' : 0,
              'bicycle' : 1,
              'bus' : 2,
              'car' : 3,
              'motorbike' : 4}

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
    for object in objects:
        bbx = object.find('bndbox')
        bbxes.append({
                      'name' : object.find('name').text,
                      'xmin' : int(float(bbx.find('xmin').text)),
                      'ymin' : int(float(bbx.find('ymin').text)),
                      'xmax' : int(float(bbx.find('xmax').text)),
                      'ymax' : int(float(bbx.find('ymax').text))
                      })
    return size, bbxes
           
def filter_boxes(target_label, bbxes):
    return filter(lambda x : x['name'] == target_label, bbxes)

def format_box(bbox):
    return '%s %.3f %d %d %d %d' % (bbox['name'],
                                    0.1,
                                    bbox['ymin'],
                                    bbox['xmin'],
                                    bbox['ymax'],
                                    bbox['xmax'])

def filter_file_list(file_list, extension = '.xml'):
    return filter(lambda x : x[-4:] == extension, file_list)

annotation_list = filter_file_list(os.listdir(annotation_folder))

output_file = open('../finetuning/VOC_windows.txt', 'wb')
crop_outputfile_train = open('../finetuning/VOC_cropped_warped_train_%s.txt' % data_set_suffix, 'wb')
crop_outputfile_test = open('../finetuning/VOC_cropped_warped_test_%s.txt' % data_set_suffix, 'wb')

idx = 0
iidx = 0
mean_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype = np.double)
#
for annotation in annotation_list:
    file_path = annotation_folder + annotation
    size, bbxes = extract_bbxes(file_path)
    bbxes = filter(lambda x : x['name'] in target_labels, bbxes)
    if len(bbxes) == 0:
        continue
    image_path = jpeg_folder + annotation[:-4] + '.jpg'
    lines_to_write = ['# ' + str(idx),
                      jpeg_folder + annotation[:-4] + '.jpg',
                      str(size['depth']),
                      str(size['height']),
                      str(size['width']),
                      str(len(bbxes))]
    lines_to_write.extend([format_box(bbx) for bbx in bbxes])
    lines_to_write.append('\n')
    output_file.write('\n'.join(lines_to_write))
    idx += 1
    #
    pict = cv2.imread(image_path)
    for bbx in bbxes:
        #
        output_path = output_cropped + ('pict_%d.jpg' % iidx)
        if np.random.uniform() > (1.0 - TEST_PROBA):
            crop_outputfile_test.write(output_path + ' ' + str(num_labels[bbx['name']]) + '\n')
        else:
            crop_outputfile_train.write(output_path + ' ' + str(num_labels[bbx['name']]) + '\n')
        iidx += 1
        if iidx > MAX_N_IMAGES:
            break
    if iidx > MAX_N_IMAGES:
        break
