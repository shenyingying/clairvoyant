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

def print_ckpt(Graph_pb):
    reader=pywrap_tensorflow.NewCheckpointReader(Graph_pb)
    var_to_shape_map=reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print(key)

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


if __name__ == '__main__':
    Graph_pb = "/home/sy/code/project/tensorflow-yolo-v3/saved_model/model.ckpt"
    # print_pb(Graph_pb)

    Graph_pb=os.path.join(Graph_pb)
    print_ckpt(Graph_pb)
