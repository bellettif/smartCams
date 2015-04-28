'''
Created on Apr 27, 2015

    Consolidate the VOC image set around cars, persons and bicycles

@author: Francois Belletti
'''

import os
import xml.etree.ElementTree as ET

main_dir            = '/Users/cusgadmin/smartCams/Wksp/smartCams/'
image_set_folder    = main_dir + 'VOC2012/ImageSets/Main/'
jpeg_folder         = main_dir + 'VOC2012/JPEGImages/'
annotation_folder   = main_dir + 'VOC2012/Annotations/'

all_sets = os.listdir(image_set_folder)

target_labels = ['person', 'bicycle', 'bus', 'car', 'motorbike']

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
        width = int(float(bbx.find('xmax').text)) - int(float(bbx.find('xmin').text))
        heigth = int(float(bbx.find('ymax').text)) - int(float(bbx.find('ymin').text))
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
                                    1.0,
                                    bbox['xmin'],
                                    bbox['ymin'],
                                    bbox['xmax'],
                                    bbox['ymax'])

def filter_file_list(file_list, extension = '.xml'):
    return filter(lambda x : x[-4:] == extension, file_list)

annotation_list = filter_file_list(os.listdir(annotation_folder))

output_file = open('VOC_windows.txt', 'wb')

idx = 0
for annotation in annotation_list:
    file_path = annotation_folder + annotation
    size, bbxes = extract_bbxes(file_path)
    bbxes = filter(lambda x : x['name'] in target_labels, bbxes)
    if len(bbxes) == 0:
        continue
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

#The window file format contains repeated blocks of:
#
#    image_index
#    img_path
#    channels
#    height 
#    width
#    num_windows
#    class_index overlap x1 y1 x2 y2
#    <... num_windows-1 more windows follow ...>    