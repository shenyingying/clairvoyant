# -*- coding: utf-8 -*-
# @Time    : 19-7-4 上午11:17
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : model_circle.py
# @Software: PyCharm
import cv2
import numpy as np
from functools import reduce


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, random=True, max_circle=4,
                    alpha=2, rotateAgle=15, proc_img=True):
    '''随机处理图像并进行数据增广'''

    line = annotation_line.split()
    image = cv2.imread(line[0])
    iw, ih = image.shape[1], image.shape[0]
    h, w = input_shape
    circle = np.array([np.array(list(map(int, circle.split(',')))) for circle in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        resize_w = int(scale * iw)
        resize_h = int(scale * ih)
        padding_w = (w - resize_w) // 2
        padding_h = (h - resize_h) // 2
        image_data = 0

        if proc_img:
            image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
            image = cv2.copyMakeBorder(image, padding_h, padding_h, padding_w, padding_w, cv2.BORDER_CONSTANT,
                                       value=(128, 128, 128))
            image_data = np.array(image) / 255.

        # correct circle
        circle_data = np.zeros((max_circle, 4))
        if (len(circle) > 0):
            np.random.shuffle(circle)
            if len(circle) > max_circle:
                circle = circle[:max_circle]  # 若lable太多 只取max_circle个
            circle[:, 0] = circle[:, 0] * scale + padding_w
            circle[:, 1] = circle[:, 1] * scale + padding_h
            circle[:, 2] = circle[:, 2] * scale
            circle_data[:len(circle)] = circle

        return image_data, circle_data

    # 实现灰度图像亮度调节,label坐标和半径不变
    alpha = rand(.5, alpha)
    blank = np.zeros(image.shape, image.dtype)
    image = cv2.addWeighted(image, alpha, blank, 1 - alpha, 0)

    # 实现图像尺寸变化,label坐标和半径都需要resize一个尺寸
    scale = rand(.1, .5)
    resize_w, resize_h = int(iw * scale), int(ih * scale)
    image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)

    # 对图片进行随机旋转操作,指定旋转的中心为图片中心,label坐标需要旋转,半径不变,
    rotateCenter = (image.shape[0] / 2, image.shape[1] / 2)
    rotateAgle = rand(-rotateAgle, rotateAgle)
    rotateMatrix = cv2.getRotationMatrix2D(rotateCenter, rotateAgle, 1)
    image = cv2.warpAffine(image, rotateMatrix, input_shape, flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    image_data = np.array(image) / 255.

    # correct circle
    circle_data = np.zeros((max_circle, 4))
    if (len(circle) > 0):
        np.random.shuffle(circle)
        if len(circle) > max_circle:
            circle = circle[:max_circle]  # 若lable太多 只取max_circle个

        # 尺度变化
        circle[:, :3] = circle[:, :3] * scale

        # 旋转
        circle_Rotate = np.transpose(circle.copy())[:3, :]
        circle_Rotate[2, :] = 1
        circle_Rotate = np.transpose(rotateMatrix.dot(circle_Rotate))
        circle[:, :2] = circle_Rotate[:, :2]

        circle_data[:len(circle)] = circle

    # for i in range(len(circle_data)):
    #     cv2.circle(image, (int(circle_data[i, 0]), int(circle_data[i, 1])), int(circle_data[i, 2]), (255, 0, 0), 1)
    # cv2.imshow('process', image)
    # cv2.waitKey()

    return image_data, circle_data


if __name__ == '__main__':
    annotation_line = "/home/sy/data/work/StandardCVSXImages/image_jpg/image_12_8.jpg 218,560,41,0 844,628,40,0"
    input_shape = (416, 416)
    get_random_data(annotation_line, input_shape, random=True)
