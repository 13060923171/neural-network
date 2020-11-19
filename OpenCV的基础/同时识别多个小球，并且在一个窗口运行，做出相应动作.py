#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import threading as th
import numpy as np

import time
import requests
import json
import concurrent.futures
from picamera import PiCamera
from picamera.array import PiRGBArray


camera = PiCamera()
camera.resolution = (450,250)
camera.framerate = 30
rawcamera = PiRGBArray(camera,size=(450,250))

left_stretch={
    "motion": {
        "direction": "left",
        "name": "stretch",
        "repeat": 1,
        "speed": "normal"
    },
    "operation": "start",
    "timestamp": int(time.time())
}

right_stretch = {
    "motion": {
        "direction": "right",
        "name": "stretch",
        "repeat": 1,
        "speed": "normal"
    },
    "operation": "start",
    "timestamp": int(time.time())
}

walk_back = {
    "motion": {
        "direction": "backward",
        "name": "walk",
        "repeat": 2,
        "speed": "normal"
    },
    "operation": "start",
    "timestamp": int(time.time())
}

wave_both = {
    "motion": {
        "direction": "both",
        "name": "wave",
        "repeat": 2,
        "speed": "normal"
    },
    "operation": "start",
    "timestamp": int(time.time())
}

reset = {
    "motion": {
        "name": "reset",
        "repeat": 1,
        "speed": "normal"
    },
    "operation": "start",
    "timestamp": int(time.time())
}

yellow_led = {
    "color": "yellow",
    "mode": "blink",
    "type": "button"
}

blue_led = {
    "color": "blue",
    "mode": "on",
    "type": "button"
}

red_led = {
    "color": "red",
    "mode": "blink",
    "type": "button"
}

tts_red = {
    "interrupt": True,
    "timestamp": int(time.time()),
    "tts": "我看到了红色小球在你左边"
}

tts_blue = {
    "interrupt": True,
    "timestamp": int(time.time()),
    "tts": "我看到蓝色小球，要躲开"
}


def r_put(model,data):
    url = 'http://127.0.0.1:9090/v1/%s' % model
    headers = {
        'Content-type': 'application/json'
    }
    data = json.dumps(data)
    response = requests.put(url,headers=headers,data=data)
    return response

low_red = np.array([173,100,100])
high_red = np.array([176,255,255])
low_blue = np.array([95,43,46])
high_blue = np.array([105,255,255])

hold_flag = 0
release_flag = 0

center_x, center_y, radius = 0,0,0
center_x2, center_y2, radius2 = 0,0,0

#获取小球的圆心半径
def get_circles(img,lower,upper):
    x,y,z = 0,0,0
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), z) = cv2.minEnclosingCircle(c)

    return int(x), int(y), int(z), img

def draw_frame(img, x, y, z):
    if (z > 15):
        cv2.circle(img, (x, y), z, (0, 255, 0), 5)
    cv2.imshow('Color_tracking', img)

def camera_thread():
    global center_x, center_y, radius,center_x2, center_y2, radius2
    for frame in camera.capture_continuous(rawcamera, format="bgr", use_video_port=True):
        image = frame.array
        x,y,z,l = get_circles(image,low_red,high_red)
        x2, y2, z2, l = get_circles(image, low_blue, high_blue)
        center_x = x
        center_y = y
        radius = z

        center_x2 = x2
        center_y2 = y2
        radius2 = z2
        draw_frame(l, x, y, z)
        draw_frame(l, x2, y2, z2)
        rawcamera.truncate(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if release_flag == 1:
            break


# 机器人动作
def red_track(x,y,z):
    if z > 20:
        if x < 150:
            r_put('devices/led',red_led)
            time.sleep(1)
            r_put('motions',right_stretch)
            time.sleep(1)
            r_put('voice/tts',tts_red)
            time.sleep(1)
        if x > 300:
            r_put('devices/led',yellow_led)
            time.sleep(1)
            r_put('motions',left_stretch)
            time.sleep(1)

def blue_track(x,y,z):
    if z > 20:
        r_put('devices/led',blue_led)
        time.sleep(1)
        r_put('motions',walk_back)
        time.sleep(2)
        r_put('motions',wave_both)
        time.sleep(2)
        r_put('voice/tts',tts_blue)
        time.sleep(1)

def track_thread():
    global center_x, center_y, radius, center_x2, center_y2, radius2
    while True:
        if radius >0:
            red_track(center_x,center_y,radius)
            time.sleep(1)
        if radius2 >0:
            blue_track(center_x2,center_y2,radius2)
            time.sleep(1)

if __name__ == '__main__':
    time.sleep(1)
    r_put('motions',reset)
    threads = []
    t1 = th.Thread(target=camera_thread, args=())
    threads.append(t1)
    t2 = th.Thread(target=track_thread, args=())
    threads.append(t2)
    for t in threads:
        t.setDaemon(True)
        t.start()
    for t in threads:
        t.join()

