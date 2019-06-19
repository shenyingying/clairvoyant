import cv2
import os
import numpy as np


RESIZE_METHOD_BILINEAR = "bilinear"
RESIZE_METHOD_ANTIALIAS = "antialias"

def get_tuned_variables():
    pass




def filter():
    # img1=cv2.imread('/home/sy/cropped_panda.jpg',cv2.IMREAD_GRAYSCALE)
    # img2=cv2.imread('/home/sy/cropped_panda.jpg',cv2.IMREAD_GRAYSCALE)
    # cv2.namedWindow("test")
    # cap = cv2.VideoCapture(0) #加载摄像头录制
    cap = cv2.VideoCapture("/home/sy/桌面/facepy/test.mp4")  # 打开视频文件
    success, frame = cap.read()
    classifier = cv2.CascadeClassifier(
        "/home/sy/桌面/facepy/haarcascade_frontalface_alt.xml")  # 确保此xml文件与该py文件在一个文件夹下，否则将这里改为绝对路径，此xml文件可在D:\My Documents\Downloads\opencv\sources\data\haarcascades下找到。

    while success:
        success, frame = cap.read()
        size = frame.shape[:2]
        image = np.zeros(size, dtype=np.float16)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(image, image)
        divisor = 8
        h, w = size
        minSize = ((int)(w / divisor), (int)(h / divisor))
        faceRects = classifier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, minSize)
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.circle(frame, ((int)(x + w / 4), (int)(y + h / 4 + 30)), min((int)(w / 8), (int)(h / 8)),
                           (255, 0, 0))
                cv2.circle(frame, ((int)(x + 3 * w / 4), (int)(y + h / 4 + 30)), min((int)(w / 8), (int)(h / 8)),
                           (255, 0, 0))
                cv2.rectangle(frame, ((int)(x + 3 * w / 8), (int)(y + 3 * h / 4)),
                              ((int)(x + 5 * w / 8), (int)(y + 7 * h / 8)), (255, 0, 0))
        cv2.imshow("test", frame)
        key = cv2.waitKey(10)
        c = chr(key & 255)
        if c in ['q', 'Q', chr(27)]:
            break
    cv2.destroyWindow("test")

    dst = cv2.ximgproc_AdaptiveManifoldFilter.filter()


def convert1024to300(file_dir, dst_dir):
    files = os.listdir(file_dir)
    for file in files:
        src_dir = file_dir + '/' + file
        src = cv2.imread(src_dir)
        zero, one = src.shape[0], src.shape[1]
        long_dim = max(zero, one)
        padding = cv2.copyMakeBorder(src, int((long_dim - zero) / 2), int((long_dim - zero) / 2),
                                     int((long_dim - one) / 2), int((long_dim - one) / 2), cv2.BORDER_CONSTANT)
        padding = cv2.resize(padding, (300, 300))

        dst = dst_dir + file
        cv2.imwrite(dst, padding)


if __name__ == '__main__':
    filter()
    # src=cv2.imread("/home/sy/data/work/snapdragon/1.jpg")
    # dst=cv2.resize(src,(1000,1000))
    # cv2.imwrite('/home/sy/data/work/snapdragon/1_1000_1000.jpg',dst)
