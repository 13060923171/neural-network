# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
import cv2
import numpy as np
import threading as th
import requests
import json
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

#初始化摄像头
camera = PiCamera()
#摄像头分辨率为450,250
camera.resolution = (640,480)
#摄像头帧数为30帧
camera.framerate = 10
#摄像头窗口大小为450,250
rawCamera = PiRGBArray(camera,size=(640,480))



#设置hsv数组
#红色数组有待改进
lower_red = np.array([173,100,100])
hight_red = np.array([176,255,255])
#黄色数组有待改进
lower_yellow = np.array([20,43,46])
hight_yellow = np.array([25,255,255])
#绿色数字正确
lower_green = np.array([65,43,46])
hight_green = np.array([80,255,255])
#蓝色数组正确
lower_blue = np.array([95,43,46])
hight_blue = np.array([105,255,255])
#紫色数组正确
lower_purple = np.array([124,43,46])
hight_purple = np.array([135,255,255])

x,y,z = 0,0,0
x2,y2,z2 = 0,0,0

#时间戳
timestamp = int(time.time())

#举左手
left_head= {
    "operation": "start",
    "motion": {
        "name": "raise",
        "direction": "left",
        "repeat": 1,
        "speed": "slow"
    },
    "timestamp": timestamp
}

#举右手
right_head= {
    "operation": "start",
    "motion": {
        "name": "raise",
        "direction": "right",
        "repeat": 1,
        "speed": "slow"
    },
    "timestamp": timestamp
}

#复原
reset= {
    "operation": "start",
    "motion": {
        "name": "reset",
        "repeat": 1,
    },
    "timestamp": timestamp
}

#亮黄灯
yellow_led={
    "type": "button",
    "color": "yellow",
    "mode": "blink"
}

#亮绿灯
green_led={
    "type": "button",
    "color": "green",
    "mode": "blink"
}

#亮蓝灯
blue_led={
    "type": "button",
    "color": "blue",
    "mode": "on"
}
#使用put方法去请求
def r_put(model,data):
    url = 'http://127.0.0.1:9090/v1/%s' % model
    headers = {
        'content-type': 'application/json'
    }
    data = json.dumps(data)
    response = requests.put(url,headers=headers,data=data)
    return response

def camera_thread():
    #调用全局变量
    global x,y,z,x2,y2,z2
    #从摄像头窗口读入数据
    for frame in camera.capture_continuous(rawCamera,format ='bgr',use_video_port=True):
        #图片为数据的数组形式
        image = frame.array
        # 高斯模糊，参数图像，卷积核，sigmaX参数一般设为0
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
        #把图片颜色转化为hsv形式
        hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
        #定义一个颜色层，并且设置它颜色的最大阈值和最小阈值
        x,y,z,image = red_camera(hsv,image)
        x2,y2,z2,image2 = yellow_camera(hsv,image)
        rawCamera.truncate(0)
        cv2.imshow('red_color', image)
        cv2.imshow('yellow_color',image2)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

def red_camera(hsv,image):
    global x,y,z
    # 生成指定颜色的遮掩层，传入图像，设置最小最大阈值
    mask = cv2.inRange(hsv, lower_red, hight_red)
    # erode腐蚀 第二参数是卷积核大小，这里不用核；腐蚀次数；
    mask = cv2.erode(mask, None, iterations=2)
    # dilate膨胀 同上
    mask = cv2.dilate(mask, None, iterations=2)
    contourareas = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contourareas) > 0:
        circle = max(contourareas, key=cv2.contourArea)
        ((x, y), z) = cv2.minEnclosingCircle(circle)
        x = int(x)
        y = int(y)
        z = int(z)
        print((x, y), 'radius:%d' % z)
    if z > 25:
        cv2.circle(image, (x, y), z, (0, 255, 0), 5)
    return x,y,z,image

def yellow_camera(hsv,image):
    global x2,y2,z2
    mask2 = cv2.inRange(hsv, lower_green, hight_green)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)
    contourareas2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(contourareas2) > 0:
        circle2 = max(contourareas2, key=cv2.contourArea)
        ((x2, y2), z2) = cv2.minEnclosingCircle(circle2)
        x2 = int(x2)
        y2 = int(y2)
        z2 = int(z2)
        print((x2, y2), 'radius:%d' % z2)
    if z2 > 25:
        cv2.circle(image, (x2, y2), z2, (0, 255, 255), 5)
    return x2,y2,z2,image

def track_thread():
    #获取全局变量
    global x,y,z,x2,y2,z2
    #写一个无线循环函数
    while True:
        #当半径大于25的时候
        if z >25:
            #当x轴的坐标小于150的时候
            if x <150:
                r_put('motions',left_head)
                time.sleep(2)
                r_put('devices/led',yellow_led)
        if z2 > 25:
            if x2 >150:
                r_put('motions',right_head)
                time.sleep(2)
                r_put('devices/led',green_led)

if __name__ == '__main__':
    #先将机器人复原
    r_put('motions',reset)
    #写两个线程
    t1 = th.Thread(target=camera_thread)
    t2 = th.Thread(target=track_thread)
    #开始线程保护
    t1.setDaemon(True)
    #开启线程
    t1.start()
    t2.setDaemon(True)
    t2.start()
    #加一个线程阻塞
    t1.join()
    t2.join()
