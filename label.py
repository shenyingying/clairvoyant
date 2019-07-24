# -*- coding: utf-8 -*-
# @Time    : 19-7-2 下午5:32
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : label.py
# @Software: PyCharm

import cv2
import numpy as np
import math
import os

drawing = False
start = (-1, 1)
radius = 0


def distance(start, end):
    return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)


# 需要把(图像路径,label信息{中心点坐标,圆半径,类别} )写到文件夹中

def mouse_event(event, x, y, flags, param):
    global point, radiuss, start, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        start = (x, y)
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        end = (x, y)
        radius = distance(start, end)
        label = str(start[0]) + "," + str(start[1]) + "," + str(math.ceil(radius)) + ',0 '
        line.append(label)
        if drawing:
            cv2.circle(img, start, math.ceil(radius), (0, 255, 0), 1)


# 读取一个文件夹下的所有.jpg文件,打开,并打label:
dst_file = "/home/sy/code/project/keras-YOLOv3-mobilenet/model_data/train_circle.txt"
pic_path = "/home/sy/data/work/StandardCVSXImages/image_jpg//"
# f = open(dst_file, 'w')
files = os.listdir(pic_path)

for file in files:
    line = []
    file = file.split('.')
    if file[-1] == 'jpg':
        # picPath = pic_path + file[0] + '.jpg'
        picPath = "/home/sy/data/work/StandardCVSXImages/image_jpg/image_3_2.jpg"
        img = cv2.imread(picPath)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', mouse_event)

    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(20)
        if (key == 27):
            labell = " "
            for i in range(len(line)):
                labell = labell + line[i]
            markInfor = picPath + labell + '\n'
            # f.writelines(markInfor)
            print(markInfor)
            break
# f.close()
