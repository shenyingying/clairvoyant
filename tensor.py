# -*- coding: utf-8 -*-
# @Time    : 19-1-31 下午4:03
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : tensor.py
# @Software: PyCharm

import os
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
from tensorflow.python import pywrap_tensorflow

'''
查看ckpt模型各个节点的名字
'''


def print_ckpt(Graph_pb):
    reader = pywrap_tensorflow.NewCheckpointReader(Graph_pb)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print(key)


'''
查看pb模型各个节点的名字
'''


def print_pb_val(Graph_pb):
    with tf.Session() as sess:
        with open(Graph_pb, 'rb')as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print(graph_def)


def print_pb(Graph_pb):
    with tf.Session() as sess:
        with gfile.FastGFile(Graph_pb, 'rb')as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
            for i, n in enumerate(graph_def.node):
                print(n.name)


'''
把.ckpt文件转化成.pb文件
'''


def convert_ckpt_pb(input_checkpoint, output_graph):
    output_node_names = 'model/block14/layer14/BiasAdd'
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(",")
        )
        with tf.gfile.GFile(output_graph, 'wb')as f:
            f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
import cv2
import numpy as np


def test_pb(pb_path):
    with tf.Graph().as_default():
        output_grapb_def = tf.GraphDef()
        with open(pb_path, 'rb') as f:
            output_grapb_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_grapb_def, name='')
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config)as sess:
                sess.run(tf.global_variables_initializer())
                input = sess.graph.get_tensor_by_name("input_1:0")
                output = sess.graph.get_tensor_by_name('conv2d_88/BiasAdd:0')

                img = cv2.imread("/home/sy/data/work/StandardCVSXImages/image_jpg/image_9_12.jpg")
                size = (320, 320)
                img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
                img.astype(np.float32)
                img = img[np.newaxis, :]
                out = sess.run(output, {input: img})
                out = np.array(out)
                print (out)


import h5py


def print_keras_weights(weight_file):
    f = h5py.File(weight_file)
    try:
        # if len(f.attrs.items()):
        #     print ("{} contains:".format(weight_file))
        #     print ('Root attributes:')
        for key, value in f.attrs.items():
            print ("{}:{}".format(key, value))
        print ('--------------------------------------')
        for layer, g in f.items():
            print (" {}".format(layer))
            print (" Attributes:")
            for key, value in g.attrs.items():
                print ("  {}: {}".format(key, value))
        #     print ("  Dataset:")
        #     for name, d in g.items():
        #         print ("  {}:{}".format(name, d.value.shape))
        #         print ("  {}:{}".format(name, d.value))
    finally:
        f.close()


if __name__ == '__main__':
    # Graph_pb = "/home/sy/code/project/tensorflow-yolo-v3/saved_model/model.ckpt"
    # Graph_pb='/home/sy/code/python/FFDnet/model/ffd/sigma/FFDnet-200'
    pb = '/home/sy/code/project/keras-YOLOv3-mobilenet/pb_result/MobileV2yolo.pb'
    # test_pb(pb)
    # print_ckpt(Graph_pb)
    print_pb(pb)
    # h5 = "/home/sy/code/project/keras-YOLOv3-mobilenet/logs/mobilenetV2/trained_weights_final.h5"
    # print_keras_weights(h5)
    # print_pb_val(pb)

    # pb_="/home/sy/code/python/FFDnet/model/ffd/FFDnet_.pb"
    # convert_ckpt_pb(Graph_pb,pb_)
    # print_pb(pb_)

    # Graph_pb=os.path.join(Graph_pb)
    # print_ckpt(Graph_pb)
