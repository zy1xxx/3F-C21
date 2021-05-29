import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *

def PerspectiveTransfer(img):
    ROTATED_SIZE  = 600 #透视变换后的表盘图像大小
    CUT_SIZE     =  0   #透视变换时四周裁剪长度

    W_cols, H_rows= img.shape[:2]
    # 选用最佳标定点取得最优效果
    pts1 = np.float32([[260, 180], [420, 180], [580, 400], [100, 400]])#[250, 180], [430, 180], [580, 400], [100, 400]
    # 变换后矩阵位置
    pts2 = np.float32([[0, 0], [ROTATED_SIZE, 0], [ROTATED_SIZE, ROTATED_SIZE], [0, ROTATED_SIZE], ])
    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst1 = cv2.warpPerspective(img, M, (ROTATED_SIZE, ROTATED_SIZE))
    return dst1


