# -*- coding: utf-8 -*-
import cv2
import numpy as np

# globe variable for control
Kp = 0.1
Ki = 0
Kd = 0
Integral_threshold = 200

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


# Incremental PID
def control(error_2, error_1, error, VR_1, VL_1):
    # 遇限削弱积分作用
    if abs(error) > Integral_threshold:
        error = 0
    delta_V = Kp * (error - error_1) + Ki * error + Kd * (error + error_2 - 2 * error_1)
    VR = VR_1 - delta_V
    VL = VL_1 + delta_V
    return VR, VL


# 输入的图像要经过透视变换
def get_error(img):
    flag=0
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Height, Width = img2.shape
    canny_img = cv2.Canny(img2, 100, 150, 5)
    _, img_bin = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
    img_bin = 255 - img_bin
    img_bin = cv2.add(img_bin, canny_img)

    # 边缘检测用于增强二值化后的图像，防止因反光而丢失赛道的情况
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bin = cv2.add(img_bin, canny_img)
    img_bin = cv2.dilate(img_bin, kernel)
    # cv2.imshow('img_bin1', img_bin)

    # 删除小面积的区域
    contours, _ = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 1500:
            cv_contours.append(contour)
        else:
            continue
    cv2.fillPoly(img_bin, cv_contours, (0, 0, 0))

    # 腐蚀和膨胀
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21))

    dilated = cv2.dilate(img_bin, kernel1)  # 第一次，膨胀用于连接可能被断开的横线
    eroded = cv2.erode(dilated, kernel2)  # 第二次，腐蚀用于消除竖向的干扰
    dilated2 = cv2.dilate(eroded, kernel3)  # 第三次，恢复上面竖向部分消除的像素，作为最终的判断图像

    # 根据连通域计算道路中点，用于区分道路和干扰的因素是：道路的外接矩形面积大或者长度大
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated2, connectivity=8)

    stats = list(stats)
    stats.sort(key=lambda x: x[4], reverse=False)
    # 去除干扰项
    if num_labels > 3:
        for i in range(num_labels - 3):
            dilated2[stats[i][1]:stats[i][1] + stats[i][3], stats[i][0]:stats[i][0] + stats[i][2]] = 0
        del stats[0:num_labels - 3]

    if len(stats) == 3:
        w1 = stats[0][2]
        h1 = stats[0][3]
        w2 = stats[1][2]
        h2 = stats[1][3]
        if w1 > Width * 4 / 5 or w2 > Width * 4 / 5:
            flag = 1
        else:
            flag = 0
        if abs(w1 * h1 - w2 * h2) > 40000:
            if w1 * h1 > w2 * h2:
                i = 1
            else:
                i = 0
            dilated2[stats[i][1]:stats[i][1] + stats[i][3], stats[i][0]:stats[i][0] + stats[i][2]] = 0
    cv2.imshow('processd', dilated2)

    # 提取预瞄点
    mid = int(Height / 2)
    _, target = np.where(dilated2[mid - 40:mid + 40, :] > 0)
    bias = float(target.mean()) - float(Width / 2)

    # bias是偏移量，flag=1是检测到了横线，flag=0是没有检测到横线
    return bias, flag


# 用于检测标志牌，0表示左转，1表示右转，-1表示没有检测到
def recong_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
    #cv2.imshow('gray',gray)
    #minRadius这个参数是根据

    circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=8, maxRadius=30)
    try:
        circles = circle1[0, :, :]  # 提取为二维
        circles = np.uint16(np.around(circles))  # 四舍五入，取整
        if len(circles) != 0:
            i = circles[0]
            x_r = i[0]
            y_r = i[1]
            radius = i[2]
            img_show=img.copy()
            cv2.circle(img_show, (i[0], i[1]), i[2], (255, 0, 0), 2)  # 画圆
            cv2.imshow('img_show',img_show)
            # cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 10)  # 画圆心

            # 目标区域的图像裁剪出来
            mask = np.ones((radius * 2, radius * 2), np.uint8) * 160
            for row in range(radius * 2):
                for col in range(radius * 2):
                    if (row - radius) ** 2 + (col - radius) ** 2 <= (radius - 2) ** 2:
                        mask[row][col] = gray[y_r - radius + row][x_r - radius + col]
            mask_size = 280
            mask = cv2.resize(mask, dsize=(mask_size, mask_size), interpolation=cv2.INTER_CUBIC)
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = 255 - mask  # 将字体变成白色
            cv2.imshow('mask',mask)
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            # print('stats is: ',stats)
            stats = list(stats)
            stats.sort(key=lambda x: x[4], reverse=False)
            # 如果找到的区域很小，认为没有找到
            if len(stats) < 2 or stats[-2][4] < (mask_size / 5) ** 2:
                print 'too small'
                return -1

            if stats[-2][2]*stats[-2][3]>(mask_size**2*4/5):
                del stats[-2]
            word_mask = mask[stats[-2][1]:stats[-2][1] + stats[-2][3], stats[-2][0]:stats[-2][0] + stats[-2][2]]

            word_size = 50
            word_mask = cv2.resize(word_mask, dsize=(word_size, word_size), interpolation=cv2.INTER_CUBIC)
            # 利用左和右的连接的差异区分
            cv2.imshow('word_mask',word_mask)

            left = word_mask[int(word_size / 3):, int(word_size / 2):int(word_size * 2 / 3)]
            right = word_mask[int(word_size / 3):, int(word_size * 2 / 3):int(word_size * 5 / 6)]
            left1 = np.where(left > 0, 1, 0)
            right1 = np.where(right > 0, 1, 0)
            num1 = int(np.sum(left1))
            num2 = int(np.sum(right1))
            # cv2.namedWindow('left',cv2.WINDOW_NORMAL)
            # cv2.namedWindow('right',cv2.WINDOW_NORMAL)
            # cv2.imshow('left',left)
            # cv2.imshow('right',right)
            # print(num1)
            # print(num2)
            if num1 > num2:
                return 0
                # return left,right,'left'
            else:
                return 1
                # return left,right,'right'
        else:
            return -1
    except:
        return -2  #没有检测到



def runCar():
    cap1 = cv2.VideoCapture(1)
    while (1):
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        _, frame1 = cap1.read()
        #frame1 = undistort(frame1)  # 图像畸变校正
        #frame1 = PerspectiveTransfer(frame1)
        turn=recong_img(frame1)
        print ('turn is:',turn)

if __name__ == '__main__':
    runCar()

