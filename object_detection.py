# -*- coding: utf-8 -*-
# @Time    : 19-1-17 下午1:17
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : trans.py
# @Software: PyCharm

import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

import time

start = time.time()
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
session=tf.Session(config=config)
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd

# if tf.__version__ < '1.4.0':
#     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

os.chdir('/home/sy/code/project/models/research/')


# Object detection imports
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# Model preparation
# What model to download.
# MODEL_NAME = 'tv_vehicle_inference_graph'
# MODEL_NAME = 'tv_vehicle_inference_graph_fasterCNN'
# MODEL_NAME = 'tv_vehicle_inference_graph_ssd_mobile'
# MODEL_NAME = '/home/sy/data/work/StandardCVSXImages/pupils'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'    #[30,21]  best
# MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'            #[42,24]
# MODEL_NAME = 'faster_rcnn_inception_v2_coco_2017_11_08'         #[58,28]
# MODEL_NAME = 'faster_rcnn_resnet50_coco_2017_11_08'     #[89,30]
# MODEL_NAME = 'faster_rcnn_resnet50_lowproposals_coco_2017_11_08'   #[64, ]
# MODEL_NAME = 'rfcn_resnet101_coco_2017_11_08'    #[106,32]

'''
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
'''

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_CKPT='/home/sy/data/work/StandardCVSXImages/log/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('training', 'tv_vehicle_detection.pbtxt')
PATH_TO_LABELS = '/home/sy/data/work/StandardCVSXImages/pupils.pbtxt'

NUM_CLASSES = 1  # 2

'''
#Download Model

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
'''

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/home/sy/data/work/StandardCVSXImages/image_jpg/'
os.chdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_DIRS = os.listdir(PATH_TO_TEST_IMAGES_DIR)

print(TEST_IMAGE_DIRS)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

output_image_path = ('/home/sy/data/work/StandardCVSXImages/ssd_result/pics/')
output_csv_path = ('/home/sy/data/work/StandardCVSXImages/ssd_result/csv/')

for image_folder in TEST_IMAGE_DIRS:
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # TEST_IMAGE_PATHS = os.listdir(os.path.join(image_folder))
            # os.makedirs(output_image_path + image_folder)
            data = pd.DataFrame()
            # for image_path in TEST_IMAGE_PATHS:
            image = Image.open(PATH_TO_TEST_IMAGES_DIR + image_folder)
            # image = Image.open(image_folder + '//'+image_path)
            width, height = image.size
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)
            # write images
            # 保存识别结果图片

            # cv2.imwrite(output_image_path+image_folder+'/'+image_path.split('\\')[-1],image_np)
            cv2.imwrite(output_image_path + image_folder, image_np)

            s_boxes = boxes[scores > 0.5]
            s_classes = classes[scores > 0.5]
            s_scores = scores[scores > 0.5]

            # write table
            # 保存位置坐标结果到 .csv表格
            for i in range(len(s_classes)):
                newdata = pd.DataFrame(0, index=range(1), columns=range(7))
                newdata.iloc[0, 0] = image_folder.split("\\")[-1].split('.')[0]
                newdata.iloc[0, 1] = s_boxes[i][0] * height  # ymin
                newdata.iloc[0, 2] = s_boxes[i][1] * width  # xmin
                newdata.iloc[0, 3] = s_boxes[i][2] * height  # ymax
                newdata.iloc[0, 4] = s_boxes[i][3] * width  # xmax
                newdata.iloc[0, 5] = s_scores[i]
                newdata.iloc[0, 6] = s_classes[i]

                data = data.append(newdata)
            data.to_csv(output_csv_path + image_folder + '.csv', index=False)

end = time.time()
print("Execution Time: ", end - start)
