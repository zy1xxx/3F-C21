#-*- coding:utf-8 -*-
import cv2

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
im_save=False
# while True:
#     _, frame1 = cap1.read()
#     _, frame2 = cap2.read()
#     cv2.imshow("image1", frame1)
#     cv2.imshow("image2", frame2)
#     cv2.waitKey(3)

#以下是修改内容，时间2021.03.15 @Author fzh
#方便关闭视频和调整窗口大小

while True:
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.imshow("image1",frame1)
    cv2.imshow("image2",frame2)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break;
    if im_save==False:
        cv2.imwrite('test.jpg',frame2)
        im_save=True

cv2.destroyAllWindows()
cap1.release()
cap2.release()