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
        ET.SubElement(root, 'path').text = '/home/sy/data/work/StandardCVSXImages/image_jpg/' + imgname + '.jpg'
        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = '1280'
        ET.SubElement(size, 'height').text = '1024'
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


if __name__ == '__main__':
    # img_dir ="/home/sy/data/work/StandardCVSXImages/image_jpg/"
    # label_txt_dir="/home/sy/data/work/StandardCVSXImages/label_txt/"
    # name_dir="/home/sy/data/work/StandardCVSXImages/classes.txt"
    # label_xml_dir="/home/sy/data/work/StandardCVSXImages/label_xml/train/"
    # txt_xml(img_dir,label_txt_dir,name_dir,label_xml_dir)

    # src_dir="/home/sy/data/work/StandardCVSXImages/image_bmp/"
    # dst_dir="/home/sy/data/work/StandardCVSXImages/im_jpg/"
    # bmp_jpg(src_dir,dst_dir,'bmp','jpg')

    src = "/home/sy/data/work/eye/label_xml/test/"
    dst = "/home/sy/data/work/eye/label_txt/test/"
    # move_format_file(src,dst,'xml')
    xml_txt(src, dst)
