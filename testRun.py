# -*- coding: utf-8 -*-  
import cv2
import math
import numpy as np
from driver import *
import time
#globe variable for control
Kp=0.8 #1
Ki=0.01 #0
Kd=1.5 #3 2.5
V=100   

#globe variable for distortionCorrect
DIM_C=(640,480)
K_C=np.array([[264.8718530264331, 0.0, 299.52281262869144],
          [0.0, 264.8614748244235, 241.52849384953572],
          [0.0, 0.0, 1.0]])
D_C=np.array([[-0.018465666357146516],
            [0.00023640864891298283],
            [0.0004638369263624063],
            [-0.0010282773327839974]])

def distortionCorrect(orinImg):
    img=undistort(orinImg)
    return img
def undistort(img, K=K_C, D=D_C, DIM=DIM_C, scale=0.6, imshow=False):
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == DIM[0] / DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
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
    ROTATED_SIZE  = 600 #透视变换后的表盘图像大小
    CUT_SIZE     =  0   #透视变换时四周裁剪长度

    W_cols, H_rows= img.shape[:2]
    # 选用最佳标定点取得最优效果
    pts1 = np.float32([[240, 220], [380, 220], [550, 400], [70, 400]])#[260, 180], [420, 180], [580, 400], [100, 400]
    # 变换后矩阵位置
    pts2 = np.float32([[0, 0], [ROTATED_SIZE, 0], [ROTATED_SIZE, ROTATED_SIZE], [0, ROTATED_SIZE], ])
    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst1 = cv2.warpPerspective(img, M, (ROTATED_SIZE, ROTATED_SIZE))
    return dst1

def control(angle1,angle2,sum):
    w=Kp*angle1+Ki*sum+Kd*(angle2-angle1)
    sum+=angle2
    VR=w/2+V
    VL=V-w/2
    print("VR,VL",(VR,VL))
    print("Sum,Dif",(sum,angle2-angle1))
    return VR,VL
def canny(img):
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)   #要二值化图像，要先进行灰度化处理
    gray=cv2.GaussianBlur(gray,(5,5),0)
    ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny_img = cv2.Canny(binary, 100, 150, 3)

    return canny_img
def slideWindow(canny_img):
    global out
    global frameCtn
    global canny_img2
    showFlag=True
    
    Pheight = canny_img.shape[0]#图片宽高
    Pwidth = canny_img.shape[1]
    width = 150#滑动窗宽高
    height = 20
    linewidthMin = 20#检测点之间的最小距离
    originPoint = (int(Pwidth / 2 - width / 2), Pheight - height)#初始框原点
    canny_img2 = canny_img.copy()#canny_img2为显示的图层

    recpointls = []#滑动窗的列表
    originAngle=0
    #确定第一个滑框的位置
    for leftRightCtn in range(3):
        position = 0
        positionCnt = 0
        cnt = 0
        poSum = 0
        line1 = False
        restart = 0
        for i in range(width):
            if line1 == False:#开始检测第一根线
                try:
                    if canny_img[originPoint[1]][originPoint[0] + i] == 255:#如果有黑色的像素点
                        cnt += 1
                        poSum = poSum + i
                        restart = i + linewidthMin#下一个开始位置
                        line1 = True
                except:
                    pass
            else:
                if i < restart:
                    continue
                else:
                    try:#检测第二根线
                        if canny_img[originPoint[1] + j][originPoint[0] + i] == 255:
                            cnt += 1
                            poSum = poSum + i
                            break
                    except:
                        pass
        if cnt != 0:
            linePo = poSum / cnt
            position = position + linePo
            positionCnt += 1
        if positionCnt==0:#说明线偏移中线
            if leftRightCtn==0:#先往右偏移
                originPoint=(originPoint[0]+width,originPoint[1])
                originAngle=0
            elif leftRightCtn==1:
                originPoint=(originPoint[0]-2*width,originPoint[1])
                originAngle=0
            else:
                print("line is too far")
                exit(0)
            # cv2.imshow("canny_img",canny_img)
        else:
            position = position / positionCnt#中线的位置
            newPoint = (int(originPoint[0] + position - width / 2), originPoint[1] - height)#下一个滑动窗的位置
            recpointls.append(originPoint)
            cv2.rectangle(canny_img2, (originPoint[0], originPoint[1]), (originPoint[0] + width, originPoint[1] + height),
                        (255, 0, 255), 2)#画出滑动窗
            originPoint = newPoint#迭代
            break
    for v in range(8):
        position = 0
        positionCnt = 0
        cnt = 0
        poSum = 0
        line1 = False
        restart = 0
        for i in range(width):
            if line1 == False:#开始检测第一根线
                try:
                    if canny_img[originPoint[1]][originPoint[0] + i] == 255:#如果有黑色的像素点
                        cnt += 1
                        poSum = poSum + i
                        restart = i + linewidthMin#下一个开始位置
                        line1 = True
                except:
                    pass
            else:
                if i < restart:
                    continue
                else:
                    try:#检测第二根线
                        if canny_img[originPoint[1] + j][originPoint[0] + i] == 255:
                            cnt += 1
                            poSum = poSum + i
                            break
                    except:
                        pass
            if cnt != 0:
                linePo = poSum / cnt
                position = position + linePo
                positionCnt += 1
      
        if positionCnt != 0:
            position = position / positionCnt#中线的位置
        newPoint = (int(originPoint[0] + position - width / 2), originPoint[1] - height)#下一个滑动窗的位置
        recpointls.append(originPoint)
        cv2.rectangle(canny_img2, (originPoint[0], originPoint[1]), (originPoint[0] + width, originPoint[1] + height),
                      (255, 0, 255), 2)#画出滑动窗
        originPoint = newPoint#迭代
    #下面开始算角度
    '''
    slopeRateSum = 0
    pointCtn = 6#取的滑动窗数量
    startPoint = 2#开始的位置
    for i in range(startPoint, pointCtn + startPoint):
        try:
            tmp = float((recpointls[i+1][0] - recpointls[i][0])) / float((recpointls[i][1] - recpointls[i+1][1]))#斜率
            slopeRateSum +=tmp
        except:
            pass
    if pointCtn!=0:
        slopeRate = slopeRateSum / pointCtn#平均斜率
        angle = math.atan(slopeRate)
        angleJ=math.degrees(angle)#角度
        print("angleJ",angleJ)
    else:
        angleJ=0

    '''
    i=0
    len=7
    slopeRate=float((recpointls[i+len][0] - recpointls[i][0])) / float((recpointls[i][1] - recpointls[i+len][1]))#斜率
    angle = math.atan(slopeRate)
    angleJ=math.degrees(angle)#角度
    print("angleJ",angleJ)
    
    if showFlag:
        cv2.imshow("canny_img", canny_img2)
    #frameCtn+=1
    return angleJ+originAngle
def getAngle(img):
    #img = distortionCorrect(img)
    img = PerspectiveTransfer(img)
    canny_img = canny(img)
    angle2= slideWindow(canny_img)
    return angle2
def runCar():
    global canny_img2
    car=driver()
    _, frame1 = cap1.read()
    angle1=getAngle(frame1)
    angleSum=angle1

    while True:
        _, frame1 = cap1.read()
        start = time.time()
        angle2=getAngle(frame1)
        angleSum+=angle2
        VR,VL=control(angle1,angle2,angleSum)
        angle1=angle2
        car.set_speed(VL,VR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canny_img2,str(int(VL))+","+str(int(VR)), (100,100), font, 0.7, (255, 255, 255), 1)
        cv2.putText(canny_img2,str(angle2), (100,200), font, 0.7, (255, 255, 255), 1)
        out.write(canny_img2)
        k = cv2.waitKey(1)
        end = time.time()
        fps=1/(end-start)
        print("fps:",fps)
        if k == ord('q'):
            break

cap1 = cv2.VideoCapture(1)
canny_img2=0
# write video
frameCtn=0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('testwrite.avi',fourcc, 2.0, (600,600),False)

runCar()
cap1.release()
out.release()
