import cv2
import os

RESIZE_METHOD_BILINEAR = "bilinear"
RESIZE_METHOD_ANTIALIAS = "antialias"


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
    file_dir = '/home/sy/data/image _source/1'
    dst_dir = file_dir + '_padding/'
    os.mkdir(dst_dir)
    convert1024to300(file_dir, dst_dir)

