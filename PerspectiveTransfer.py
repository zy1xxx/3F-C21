# #（基于透视的图像矫正）
# import cv2
# import math
# import numpy as np
#
# def Img_Outline(input_dir):
#     original_img = cv2.imread(input_dir)
#     gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray_img, (1, 1), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
#     _, RedThresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))          # 定义矩形结构元素
#     closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
#     opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
#     return original_img, gray_img, RedThresh, closed, opened
#
#
# def findContours_img(original_img, opened):
#     contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     #c = sorted(contours, key=cv2.contourArea, reverse=True)[1]   # 计算最大轮廓的旋转包围盒
#     #rect = cv2.minAreaRect(c)                                    # 获取包围盒（中心点，宽高，旋转角度）
#     #box = np.int0(cv2.boxPoints(rect))                           # box
#     #box[]
#     #draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
#     draw_img=original_img.copy()
#     cv2.drawContours(draw_img, contours, -1, (255, 0, 0), 2)
#
#     #拟合四边形
#     #计算封闭轮廓周长
#     cnt_len = cv2.arcLength(contours[0], True)
#     #
#     box = cv2.approxPolyDP(contours[0], 0.02 * cnt_len, True)
#     if len(box) == 4:
#         cv2.drawContours(draw_img, [box], -1, (255, 255, 0), 3)
#
#     '''
#     box[0]: [[163  32]]右上
#     box[1]: [[63 72]]   左上
#     box[2]: [[150 215]]左下
#     box[3]: [[268 144]]右下
#     '''
#
#     print("box[0]:", box[0])
#     print("box[1]:", box[1])
#     print("box[2]:", box[2])
#     print("box[3]:", box[3])
#     # for i in range(len(box)):
#     #     box_after[i]=box[3-i]
#     box_after =[0]*4
#     #排好序的角点输出，0号是左上角，顺时针输出
#     box_after[0] = box[1]
#     box_after[1] = box[0]
#     box_after[2] = box[3]
#     box_after[3] = box[2]
#     print("box_after[0]:", box_after[0])
#     print("box_after[1]:", box_after[1])
#     print("box_after[2]:", box_after[2])
#     print("box_after[3]:", box_after[3])
#     return box_after,draw_img
#     #return draw_img
# def Perspective_transform(box,original_img):
#     # # 获取画框宽高(x=orignal_W,y=orignal_H)
#     # orignal_W = math.ceil(np.sqrt((box[3][1] - box[2][1])**2 + (box[3][0] - box[2][0])**2))
#     # orignal_H= math.ceil(np.sqrt((box[3][1] - box[0][1])**2 + (box[3][0] - box[0][0])**2))
#     #
#     # # 原图中的四个顶点,与变换矩阵
#     # pts1 = np.float32([box[0], box[1], box[2], box[3]])
#     # pts2 = np.float32([[int(orignal_W+1),int(orignal_H+1)], [0, int(orignal_H+1)], [0, 0], [int(orignal_W+1), 0]])
#     #
#     # # 生成透视变换矩阵；进行透视变换
#     # M = cv2.getPerspectiveTransform(pts1, pts2)
#     # result_img = cv2.warpPerspective(original_img, M, (int(orignal_W+3),int(orignal_H+1)))
#     #
#
#     ROTATED_SIZE_W = 600  # 透视变换后的表盘图像大小
#     ROTATED_SIZE_H = 800  # 透视变换后的表盘图像大小
#     # 原图中书本的四个角点(左上、右上、右下、左下),与变换后矩阵位置
#     #pts1 = np.float32([[63, 72], [163, 32], [268, 144], [150, 215]])
#     pts1 = np.float32([box[0], box[1], box[2], box[3]])
#     # 变换后矩阵位置
#     pts2 = np.float32([[0, 0], [ROTATED_SIZE_W, 0], [ROTATED_SIZE_W, ROTATED_SIZE_H], [0, ROTATED_SIZE_H], ])
#     # 生成透视变换矩阵；进行透视变换
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     result_img = cv2.warpPerspective(original_img, M, (ROTATED_SIZE_W, ROTATED_SIZE_H))
#
#
#     return result_img
#
# if __name__=="__main__":
#     input_dir = "PT_test1.jpg"
#     original_img, gray_img, RedThresh, closed, opened = Img_Outline(input_dir)
#     # box, draw_img = findContours_img(original_img,opened)
#     # # draw_img = findContours_img(original_img, opened)
#     # result_img = Perspective_transform(box,original_img)
#     contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     draw_img=original_img.copy()
#     cv2.drawContours(draw_img, contours, -1, (255, 0, 0), 2)
#     cv2.imshow("original", original_img)
#     cv2.imshow("gray", gray_img)
#     cv2.imshow("closed", closed)
#     cv2.imshow("opened", opened)
#     cv2.imshow("draw_img", draw_img)
#     # cv2.imshow("result_img", result_img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#

'''
box[0]: [[163  32]]右上
box[1]: [[63 72]]   左上
box[2]: [[150 215]]左下
box[3]: [[268 144]]右下
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *

# im = array(Image.open('PT_test1.jpg'))
# imshow(im)
# show()
img1 = cv2.imread('CorImage1.png')

ROTATED_SIZE  = 600 #透视变换后的表盘图像大小
CUT_SIZE     =  0   #透视变换时四周裁剪长度

W_cols, H_rows= img1.shape[:2]
print(H_rows, W_cols)

# 原图中书本的四个角点(左上、右上、右下、左下),与变换后矩阵位置,排好序的角点输出，0号是左上角，顺时针输出
pts1 = np.float32([[252, 178], [345, 179], [432, 311], [154, 306]])
#变换后矩阵位置
pts2 = np.float32([[0, 0],[ROTATED_SIZE,0],[ROTATED_SIZE, ROTATED_SIZE],[0,ROTATED_SIZE],])


# 生成透视变换矩阵；进行透视变换
M = cv2.getPerspectiveTransform(pts1, pts2)
dst1 = cv2.warpPerspective(img1, M, (ROTATED_SIZE,ROTATED_SIZE))

# im=array(img)
# imshow(im)
# show()
cv2.imshow("result",dst1)
# show()
cv2.waitKey(0)
cv2.destroyAllWindows()

