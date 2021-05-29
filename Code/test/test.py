import cv2
import numpy as np
#读取图像
img = cv2.imread("00.png")
#将原图转为灰度图
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Canny边缘检测
canny_img = cv2.Canny(gray_img,100,150,3)
#显示边缘检测后的图像
cv2.imshow("canny_img",canny_img)
cv2.imwrite("out00.png",canny_img)
cv2.waitKey(0)
