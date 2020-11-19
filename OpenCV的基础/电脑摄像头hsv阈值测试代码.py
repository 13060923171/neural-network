import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret = cap.set(3, 640)  # 设置帧宽
ret = cap.set(4, 480)  # 设置帧高


def camera():  # 检查摄像头是否正常启动
    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 转换为HSV空间
        lower_green = np.array([65, 43, 46])  # 设定绿色的阈值下限
        upper_green = np.array([80, 255, 255])  # 设定绿色的阈值上限
        lower_blue = np.array([95, 43, 46])  # 设定绿色的阈值下限
        upper_blue = np.array([105, 255, 255])  # 设定绿色的阈值上限
        lower_red = np.array([2, 100, 100])  # 设定绿色的阈值下限
        upper_red = np.array([4, 255, 255])  # 设定绿色的阈值上限
        lower_yellow = np.array([20, 43, 46])  # 设定绿色的阈值下限
        upper_yellow = np.array([25, 255, 255])  # 设定绿色的阈值上限
        lower_purple = np.array([124, 43, 46])  # 设定绿色的阈值下限
        upper_purple = np.array([135, 255, 255])  # 设定绿色的阈值上限
        mask = cv2.inRange(hsv,lower_green,upper_green)
        mask2 = cv2.inRange(hsv,lower_blue,upper_blue)
        mask3 = cv2.inRange(hsv, lower_purple, upper_purple)
        mask4 = cv2.inRange(hsv, lower_red, upper_red)
        mask5 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        contourareas = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contourareas2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contourareas3 = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contourareas4 = cv2.findContours(mask4.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contourareas5 = cv2.findContours(mask5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(contourareas) >0:
            circle = max(contourareas,key=cv2.contourArea)
            ((x,y),z) = cv2.minEnclosingCircle(circle)
            x = int(x)
            y = int(y)
            z = int(z)
            print((x,y),'radius:%d' % z)
            if z >25:
                cv2.circle(frame,(x,y),z,(0,255,0),5)
        if len(contourareas2) >0:
            circle2 = max(contourareas2,key=cv2.contourArea)
            ((x2,y2),z2) = cv2.minEnclosingCircle(circle2)
            x2 = int(x2)
            y2 = int(y2)
            z2 = int(z2)
            print((x2,y2),'radius:%d' % z2)
            if z2 >25:
                cv2.circle(frame,(x2,y2),z2,(0,255,255),5)
        if len(contourareas3) >0:
            circle3 = max(contourareas3,key=cv2.contourArea)
            ((x3,y3),z3) = cv2.minEnclosingCircle(circle3)
            x3 = int(x3)
            y3 = int(y3)
            z3 = int(z3)
            print((x3,y3),'radius:%d' % z3)
            if z3 >25:
                cv2.circle(frame,(x3,y3),z3,(255,255,255),5)

        if len(contourareas4) >0:
            circle4 = max(contourareas4,key=cv2.contourArea)
            ((x4,y4),z4) = cv2.minEnclosingCircle(circle4)
            x4 = int(x4)
            y4 = int(y4)
            z4 = int(z4)
            print((x4,y4),'radius:%d' % z4)
            if z4 >25:
                cv2.circle(frame,(x4,y4),z4,(0,0,0),5)
        if len(contourareas5) >0:
            circle5 = max(contourareas5,key=cv2.contourArea)
            ((x5,y5),z5) = cv2.minEnclosingCircle(circle5)
            x5 = int(x5)
            y5 = int(y5)
            z5 = int(z5)
            print((x5,y5),'radius:%d' % z5)
            if z5 >25:
                cv2.circle(frame,(x5,y5),z5,(255,0,0),5)
        cv2.imshow('camera',frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

if __name__ == '__main__':

    camera()
