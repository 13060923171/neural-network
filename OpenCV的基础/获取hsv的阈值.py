# import numpy as np
# import cv2
# green = np.uint8([[[0,255,0 ]]])
# hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
# print( hsv_green )


import cv2
import numpy as np
from matplotlib import pyplot as plt

# cap = cv2.VideoCapture(0)
# while(1):
#     # 获得图片
#     ret, frame = cap.read()
#     # 展示图片
#     cv2.imshow("capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         # 存储图片
#         cv2.imwrite("red.jpg", frame)
#         break


image=cv2.imread('red.jpg')
HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
def getpos(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN: #定义一个鼠标左键按下去的事件
        print(HSV[y,x])

cv2.imshow("imageHSV",HSV)
cv2.imshow('image',image)
cv2.setMouseCallback("imageHSV",getpos)
cv2.waitKey(0)

