# -*- coding: utf-8 -*-
# @Time    : 19-6-26 下午5:14
# @Author  : shenyingying
# @Email   : shen222ying@163.com
# @File    : draw_circle_label.py
# @Software: PyCharm

import cv2
import numpy as np
import math


def on_EVENT_LBUTTONDOWN(event1, event2, centerX, centerY, flags, param):
    n = 0
    circle = []
    if (n == 0):
        if event1 == cv2.EVENT_LBUTTONDOWN:
            firstPoint = "%d,%d" % (centerX, centerY)
            circle.append(centerX)
            circle.append(centerY)
            # if event == cv2.EVENT_RBUTTONDOWN:
            #     secondPoint = "%d,%d" % (radiusX, radiusY)
            #     r = math.sqrt((centerX - radiusX) ** 2 + (centerY - radiusY) ** 2)
            # print(centerX, centerY)
            n = 1
    else:
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # secondPoint = "%d,%d" % (centerX, centerY)
            cv2.circle(img, (circle[0], circle[1]), 100, (255, 0, 0), thickness=1)
            # cv2.putText(img, firstPoint, (centerX, centerY), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)


img = cv2.imread("/home/sy/data/work/StandardCVSXImages/image_jpg/image_1_12.jpg")
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()

# if __name__ == '__main__':
#     cv2.waitKey()
