# -*- coding: utf-8 -*-  
import cv2
import math
import numpy as np
from driver import *

# globe variable for control
Kp = 0.5
Ki = 0
Kd = 0
Integral_threshold=45
wheelBase = 15
V = 30

# globe variable for image revise
DIM_C = (640, 480)
K_C = np.array([[264.8718530264331, 0.0, 299.52281262869144],
                [0.0, 264.8614748244235, 241.52849384953572],
                [0.0, 0.0, 1.0]])
D_C = np.array([[-0.018465666357146516],
                [0.00023640864891298283],
                [0.0004638369263624063],
                [-0.0010282773327839974]])

def undistort(img, K=K_C, D=D_C, DIM=DIM_C, scale=0.6, imshow=False):
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == DIM[0] / DIM[
        1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0] != DIM[0]:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:  # change fov
        Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
    return undistorted_img


def PerspectiveTransfer(img):
    ROTATED_SIZE = 600  # 透视变换后的表盘图像大小
    CUT_SIZE = 0  # 透视变换时四周裁剪长度

    W_cols, H_rows = img.shape[:2]
    # 选用最佳标定点取得最优效果
    pts1 = np.float32([[240, 220], [380, 220], [550, 400], [70, 400]])  # [260, 180], [420, 180], [580, 400], [100, 400]
    # 变换后矩阵位置
    pts2 = np.float32([[0, 0], [ROTATED_SIZE, 0], [ROTATED_SIZE, ROTATED_SIZE], [0, ROTATED_SIZE], ])
    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst1 = cv2.warpPerspective(img, M, (ROTATED_SIZE, ROTATED_SIZE))
    return dst1


#Incremental PID
def control(anglek_2, anglek_1, anglek):
    # 遇限削弱积分作用
    if abs(anglek)>Integral_threshold:
        anglek=0
    delta_V=Kp*(anglek-anglek_1)+Ki*anglek+Kd*(anglek+anglek_2-2*anglek_1)
    VR = V + delta_V* wheelBase / 2
    VL = V - delta_V * wheelBase / 2
    return VR, VL


def canny(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 100, 150, 3)
    return canny_img


def slideWindow(canny_img):
    Pheight = canny_img.shape[0]
    Pwidth = canny_img.shape[1]
    # print(canny_img.shape)
    width = 200
    height = 40
    linewidthMin = 20
    originPoint = (int(Pwidth / 2 - width / 2), Pheight - height)
    canny_img2 = canny_img.copy()
    cv2.rectangle(canny_img2, (originPoint[0], originPoint[1]), (originPoint[0] + width, originPoint[1] + height),
                  (255, 0, 255), 2)
    cv2.imshow("canny_img", canny_img2)
    recpointls = []
    recpointls.append(originPoint)
    for v in range(int(Pheight / height)):
        # print("rec index:", v)
        # print(originPoint)
        position = 0
        positionCnt = 0
        for j in range(3):
            cnt = 0
            poSum = 0
            line1 = False
            restart = 0
            for i in range(width):
                if line1 == False:
                    try:
                        if canny_img[originPoint[1] + j][originPoint[0] + i] == 255:
                            cnt += 1
                            poSum = poSum + i
                            restart = i + linewidthMin
                            line1 = True
                            # print(i)
                    except:
                        pass
                else:
                    if i < restart:
                        continue
                    else:
                        try:
                            if canny_img[originPoint[1] + j][originPoint[0] + i] == 255:
                                cnt += 1
                                poSum = poSum + i
                                # print(i)
                                break
                        except:
                            pass
            if cnt != 0:
                linePo = poSum / cnt
                # print('cnt:',cnt)
                position = position + linePo
                positionCnt += 1
        if positionCnt != 0:
            position = position / positionCnt
        # print("aver:", position)
        # print("<<<<<<")
        newPoint = (int(originPoint[0] + position - width / 2), originPoint[1] - height)
        recpointls.append(newPoint)
        cv2.rectangle(canny_img2, (newPoint[0], newPoint[1]), (newPoint[0] + width, newPoint[1] + height),
                      (255, 0, 255), 2)
        originPoint = newPoint
    slopeRateSum = 0
    pointCtn = 5
    startPoint = 1
    angle = 10

    for i in range(startPoint, pointCtn + startPoint):
        try:
            tmp = (recpointls[i + 1][1] - recpointls[i][1]) / (recpointls[i][0] - recpointls[i + 1][0])
            slopeRateSum = +tmp
        except:
            pass
    if pointCtn != 0:
        slopeRate = slopeRateSum / pointCtn
        # print(slopeRate)
        angle = math.atan(slopeRate)
        print("angle", 90 - math.degrees(angle))
    else:
        angle = 0
    cv2.imshow("canny_img", canny_img2)

    return angle


def getAngle(img):
    #img 是已经校正过的图像
    img = PerspectiveTransfer(img)
    canny_img = canny(img)
    angle2 = slideWindow(canny_img)
    return angle2

def runCar():
    car = driver()
    cap1 = cv2.VideoCapture(1)
    anglek_1=0
    anglek_2=0
    while True:
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        _, frame1 = cap1.read()
        frame1 = undistort(frame1)

        # img=undistort(frame1)
        # rightimg=PerspectiveTransfer(img)
        # canny_img = canny(img)
        # cv2.imshow("canny_img", canny_img)

        anglek = getAngle(frame1)
        VR, VL = control(anglek, anglek_1,anglek_2)
        anglek_2=anglek_1
        anglek_1=anglek

        car.set_speed(VR, VL)


if __name__ == '__main__':
    runCar()

