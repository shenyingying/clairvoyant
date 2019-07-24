# -*- coding: utf-8 -*-
# @Time    : 19-1-22 下午4:24
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : transs.py
# @Software: PyCharm
import os
import cv2
import copy
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import shutil


# move one format file to another file
def move_file(src, dst, format):
    files = os.listdir(src)
    for file in files:
        src_name = src + '/' + file
        dst_name = dst + '/' + file
        file = file.split('.')
        if file[-1] == format:
            shutil.move(src_name, dst_name)


# convert pic with src_format to dst_format
def bmp_jpg(src_dir, dst_dir, src_format, dst_format):
    files = os.listdir(src_dir)
    for file in files:
        file = file.split('.')
        if file[-1] == src_format:
            # 把要读图片连接成一个路径
            src_file_name = src_dir + '/' + file[0] + '.' + file[-1]
            img = cv2.imread(src_file_name)
            dst_file_name = dst_dir + '/' + file[0] + '.' + dst_format
            # 需要waitkey，否则转换不成功
            cv2.waitKey(1)
            cv2.imwrite(dst_file_name, img)


def change_content(label_txt_dir):
    labels = os.listdir(label_txt_dir)
    for label in labels:
        with open(label, 'r')as f:
            label = f.readlines()
            label = [lab.strip('\n') for lab in label]
            lab = lab.split()


# convert label with .txt to .xml
def txt_xml(img_dir, label_txt_dir, name_dir, label_xml_dir):
    labels = os.listdir(label_txt_dir)
    with open(name_dir, 'r')as f:
        classes = f.readlines()
        classes = [cls.strip('\n') for cls in classes]

    def write_xml(imgname, filepath, labeldicts):
        root = ET.Element('annotation')
        ET.SubElement(root, 'folder').text = 'image'
        ET.SubElement(root, 'filename').text = imgname + '.jpg'
        ET.SubElement(root, 'path').text = '/home/sy/data/work/eye/image_jpg/' + imgname + '.jpg'
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = '700'
        ET.SubElement(size, 'height').text = '605'
        ET.SubElement(size, 'depth').text = '3'
        ET.SubElement(root, 'segmented').text = '0'
        for labeldict in labeldicts:
            object = ET.SubElement(root, 'object')
            ET.SubElement(object, 'name').text = labeldict['name']
            ET.SubElement(object, 'pose').text = 'Unspecified'
            ET.SubElement(object, 'truncated').text = '0'
            ET.SubElement(object, 'difficult').text = '0'
            bndbox = ET.SubElement(object, 'bndbox')
            # xml 用最小点和最大点表示矩形
            ET.SubElement(bndbox, 'xmin').text = str(int(labeldict['xmin']))
            ET.SubElement(bndbox, 'ymin').text = str(int(labeldict['ymin']))
            ET.SubElement(bndbox, 'xmax').text = str(int(labeldict['xmax']))
            ET.SubElement(bndbox, 'ymax').text = str(int(labeldict['ymax']))
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8')

    for label in labels:
        with open(label_txt_dir + label, 'r')as f:
            # img_id=os.path.split(label)[0]
            img_id = label.split('.')[0]
            contents = f.readlines()
            labeldicts = []
            for content in contents:
                img = np.array(Image.open(img_dir + label.strip('.txt') + '.jpg'))
                sh, sw = img.shape[0], img.shape[1]
                content = content.strip('\n').split()
                # txt 用中心点和长宽表示矩形
                x = float(content[1]) * sw
                y = float(content[2]) * sh
                w = float(content[3]) * sw
                h = float(content[4]) * sh
                new_dict = {'name': classes[int(content[0])],
                            'difficult': '0',
                            'xmin': x + 1 - w / 2,
                            'ymin': y + 1 - h / 2,
                            'xmax': x + 1 + w / 2,
                            'ymax': y + 1 + h / 2}
                labeldicts.append(new_dict)
                write_xml(img_id, label_xml_dir + label.strip('.txt') + '.xml', labeldicts)


def remove_format_file(path, format):
    files = os.listdir(path)
    for file in files:
        file = file.split('.')
        if (file[-1] == format):
            os.remove(path + file[0] + '.' + format)
        else:
            continue


def move_format_file(src, dst, format):
    files = os.listdir(src)
    for file in files:
        file = file.split('.')
        if (file[-1] == format):
            src_file = src + file[0] + '.' + file[1]
            dst_file = dst + file[0] + '.' + format
            shutil.move(src_file, dst_file)
        else:
            continue


import glob


def xml_txt(src, dst):
    try:
        import xml.etree.cElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET

    files = os.listdir(src)
    for xml_file in files:
        tree = ET.parse(src + xml_file)
        filename = xml_file.split('.')[0] + '.txt'
        f = open(dst + filename, 'w')
        root = tree.getroot()
        for member in root.findall('object'):
            width = float(root.find('size')[0].text)
            heith = float(root.find('size')[1].text)
            xmin = float(member[4][0].text)
            ymin = float(member[4][1].text)
            xmax = float(member[4][2].text)
            ymax = float(member[4][3].text)

            xmid = round((xmin + xmax) / (2 * width), 8)
            ymid = round((ymin + ymax) / (2 * heith), 8)
            x_width = round((xmax - xmin) / width, 8)
            y_height = round((ymax - ymin) / heith, 8)
            content = '0 ' + str(xmid) + ' ' + str(ymid) + ' ' + str(x_width) + ' ' + str(y_height)
            print(content)

        f.write(content)
        f.close()


def generate_train(pic_path, dst_file):
    f = open(dst_file, 'w')
    files = os.listdir(pic_path)
    for file in files:
        file = file.split('.')
        if file[-1] == 'jpg':
            line = pic_path + file[0] + '.jpg\n'
            f.writelines(line)
    f.close()


def test(pic):
    mat = cv2.imread(pic)
    out1 = mat[..., [2, 1, 0]]
    # cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    cv2.imshow(" ", out1)
    cv2.waitKey()


import cv2


def raw_pic(pic):
    img = np.ones((2, 4), dtype=np.uint8)
    img[0, 0] = 100
    img[0, 1] = 150
    img[0, 2] = 255
    cv2.imshow("img", img)
    cv2.waitKey()
    brg_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow("brg_img", brg_img)
    cv2.waitKey()

    print(img)
    print("brg_hwc:")
    print(brg_img)
    print(brg_img.shape[0], brg_img.shape[1], brg_img.shape[2])

    print("brg_img_chw:")
    brg_img_chw = np.transpose(brg_img, (2, 0, 1))
    print(brg_img_chw)
    print(brg_img_chw.shape[0], brg_img_chw.shape[1], brg_img_chw.shape[2])

    mat_pic_dim = (3, 2, 2)
    mat_dim = (2, 2)

    mat_r = np.empty(mat_dim)
    mat_r.fill(1)
    mat_g = np.empty(mat_dim)
    mat_g.fill(2)
    mat_b = np.empty(mat_dim)
    mat_b.fill(3)

    mat = np.empty(mat_pic_dim)

    mat[0] = mat_r
    mat[1] = mat_g
    mat[2] = mat_b
    print(mat)

    mat = np.transpose(mat, (1, 2, 0))
    print(mat)

    # print mat_b
    # print mat_g
    # print mat_r
    # mat={1,2,3,4,5,6,7,8,9,10,11,12}
    #
    mat = cv2.imread(pic)
    print(mat.shape[0], mat.shape[1], mat.shape[2])
    # print (mat)


import numpy as np
import os


def np_file():
    a = np.arange(0, 12)
    print(a)
    a.reshape(3, 4)
    print("a:")
    print(a)
    a.tofile("a.raw")
    b = np.fromfile("a.raw", dtype=np.int)
    print("b:")
    print(b)
    b.reshape(3, 4)
    print(b)

    np.float
    np.float32


def generate_empty_lable(src):
    files = os.listdir(src)
    for file in files:
        file = file.split('.')
        txt_file_path = src + file[0] + '.txt'
        txt_file = open(txt_file_path, 'w')
        txt_file.write(' ')


def video_pic(path):
    cap = cv2.VideoCapture(path)
    sucess = cap.isOpened()
    frame_count = 0
    i = 0
    while sucess:
        frame_count += 1
        sucess, frame = cap.read()
        if (frame_count % 10 == 0):
            i += 1
            cv2.imwrite('/home/sy/s/d%d.jpg' % i, frame)
    cap.release()


def convert_tif_to_png(src):
    files = os.listdir(src)
    for file in files:
        src_pic = src + file
        img = cv2.imread(src_pic)
        name = file.split('.')
        name = src + name[0] + '.png'
        cv2.imwrite(name, img)
        cv2.waitKey(1)


def convert_gif_to_png(src):
    files = os.listdir(src)
    for file in files:
        src_pic = src + file
        gif = cv2.VideoCapture(src_pic)
        ret, img = gif.read()
        name = file.split('.')
        name = src + name[0] + '.png'
        cv2.imwrite(name, img)
        cv2.waitKey(1)


# rename file
def change_file_name(dir):
    files = os.listdir(dir)
    for file in files:
        # print (file[9:])
        if (file[0] == 'l'):
            src_name = dir + file
            dst_name = dir + file[9:]
            os.rename(src_name, dst_name)


def read(src_txt):
    f = open(src_txt, 'r')
    try:
        for line in f:
            print(line)
    finally:
        f.close()



def mouse_event(event, x, y, flags, param):
    global start, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


def draw_circle_lable():
    img = cv2.imread("/home/sy/data/work/StandardCVSXImages/image_jpg/image_1_12.jpg")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_event)
    while True:
        cv2.imshow('image', img)
        if cv2.waitKey(20) == 27:
            break


if __name__ == '__main__':
    src = "/home/sy/code/git/cvs-mobilenetv2-yolo/resource/"
    dst = "/home/sy/code/git/cvs-mobilenetv2-yolo/resource/15_7-12/"
    format = "jpg"

    # src = "/home/sy/data/work/StandardCVSXImages/image_jpg/"
    # dst = "/home/sy/code/project/mAP/inputs_your/MobileNetV2_0.5_224_layerShort/"
    # format = "txt"
    move_format_file(src, dst, format)

    # draw_circle_lable()
