# -*- coding: utf-8 -*-
# @Time    : 19-4-19 下午3:02
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : vision.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

import os


# import cv2


def print_pb():
    graph_path = '/home/sy/code/project/DW2TF/data/yolov3.pb'
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        graph_def = tf.GraphDef()
        with open(graph_path, 'rb')as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            for i, n in enumerate(graph_def.node):
                print ('Name of the node - %s' % n.name)


def print_ckpt():
    ckpt_path = "/home/sy/code/project/DW2TF/data/yolov3.ckpt"
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print ("tensor_name: ", key)


def get_tuned_variables():
    CKPTPOINT_EXCLUDE_SCOPES = 'CDnCNN/block18,CDnCNN/block16'
    exclusions = [scope.strip() for scope in CKPTPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
            if not excluded:
                variables_to_restore.append(var)
    return variables_to_restore


def delete_graph():
    load_fn = slim.assign_from_checkpoint_fn(
        model_path='/home/sy/code/python/FFDnet/model/mnist/mnist.ckpt',
        var_list=get_tuned_variables(),
        ignore_missing_vars=True
    )
    saver = tf.train.Saver()
    with tf.Session()as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        load_fn(sess)


# def video_pic(src, dst):
#     f = 0
#     idxName = 0
#     video = cv2.VideoCapture(src)
#     if video.isOpened():
#         rval, frame = video.read()
#     else:
#         rval = False
#
#     while rval:
#         rval, frame = video.read()
#         if (f % 10 == 0):
#             cv2.imwrite(dst + str(idxName) + '.jpg', frame)
#             idxName = idxName + 1
#         f = f + 1
#         cv2.waitKey(1)
#     video.release()


if __name__ == '__main__':
    # delete_graph()
    print_pb()
    print ("----------------------------------------------------")
    # print_ckpt()
