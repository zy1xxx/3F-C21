# -*- coding: utf-8 -*-  
import cv2
import math
import numpy as np
from driver import *
import time
#globe variable for control
Kp=0.5
Ki=0
Kd=0
wheelBase=15
V=30

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
    return VR,VL
def canny(img):
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)   #要二值化图像，要先进行灰度化处理
    ret, binary_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    canny_img = cv2.Canny(binary_img, 100, 150, 3)
    return canny_img,binary_img
def slideWindow(canny_img,binary_img):
    global out
    global frameCtn
    showFlag=True
    
    Pheight = canny_img.shape[0]#图片宽高
    Pwidth = canny_img.shape[1]
    width = 100#滑动窗宽高
    height = 20
    linewidthMin = 20#检测点之间的最小距离
    originPoint = (int(Pwidth / 2 - width / 2), Pheight - height)#初始框原点
    canny_img2 = canny_img.copy()#canny_img2为显示的图层
    cv2.rectangle(canny_img2, (originPoint[0], originPoint[1]), (originPoint[0] + width, originPoint[1] + height),
                  (255, 0, 255), 2)
    cv2.rectangle(binary_img, (originPoint[0], originPoint[1]), (originPoint[0] + width, originPoint[1] + height),
                  (255, 0, 255), 2)
    if showFlag:
        cv2.imshow("canny_img", canny_img2)
        cv2.imshow("binary_img", binary_img)
    recpointls = []#滑动窗的列表
    recpointls.append(originPoint)
    for v in range(int(Pheight / height)):
        position = 0
        positionCnt = 0
        for j in range(3):#检测的个数
            cnt = 0
            poSum = 0
            line1 = False
            restart = 0
            for i in range(width):
                if line1 == False:#开始检测第一根线
                    try:
                        if canny_img[originPoint[1] + j][originPoint[0] + i] == 255:#如果有黑色的像素点
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
        recpointls.append(newPoint)
        cv2.rectangle(canny_img2, (newPoint[0], newPoint[1]), (newPoint[0] + width, newPoint[1] + height),
                      (255, 0, 255), 2)#画出滑动窗
        cv2.rectangle(binary_img, (newPoint[0], newPoint[1]), (newPoint[0] + width, newPoint[1] + height),
                      (255, 0, 255), 2)#画出滑动窗
        originPoint = newPoint#迭代
    #下面开始算角度
    slopeRateSum = 0
    pointCtn = 7#取的滑动窗数量
    startPoint = 3#开始的位置
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
    if showFlag:
        cv2.imshow("canny_img", canny_img2)
        cv2.imshow("binary_img", binary_img)
    #下面检测十字线
    global turnFlag
    start=7
    blackPointCnt=0
    for i in range(height):
        for j in range(width):
            if (binary_img[recpointls[start][0]+j][recpointls[start][1]+i]==0):
                blackPointCnt+=1
    blackRate=float(blackPointCnt)/(height*width)
    print('blackRate',blackRate)
    if blackRate>0.9:
        turnFlag=True
        return 0
    #out.write(canny_img2)
    #frameCtn+=1
    return angleJ
def getAngle(frame1):
    img = distortionCorrect(frame1)
    img = PerspectiveTransfer(img)
    canny_img,binary_img = canny(img)
    angle2 = slideWindow(canny_img,binary_img)
    return angle2
def turn(d):
    start = time.time()
    runTime=3*1000#3秒
    V=30
    if d==0:#right
        car.set_speed(V,-V)
    else:
        car.set_speed(-V,V)
    while time.time()-start>runTime:
        break
    return 
def runTurn(d):
    global turnFlag
    start = time.time()
    V=30
    runTime=3*1000#3秒
    car.set_speed(V,V)
    while time.time()-start>runTime:
        break
    turn(d)
    turnFlag=False
def getD():
    car.set_speed(0,0)
    time.sleep(3000)
    return 0
def runCar():
    turnFlag=False
    car=driver()
    _, frame1 = cap1.read()
    angle1=getAngle(frame1)
    angleSum=angle1
    while True:
        _, frame1 = cap1.read()
        #start = time.time()
        angle2=getAngle(frame1)
        if turnFlag:
            pass
            #runTurn(getD())
        #end = time.time()
        #fps=1/(end-start)
        #print("fps:",fps)
        else:
            angleSum+=angle2
            VR,VL=control(angle1,angle2,angleSum)
            angle1=angle2
            car.set_speed(VL,VR)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

cap1 = cv2.VideoCapture(1)
'''write video
frameCtn=0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('testwrite.avi',fourcc, 2.0, (600,600),False)
'''
runCar()
cap1.release()
#out.release()
