#-*- coding:utf-8 -*-
import cv2
import numpy as np
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
im_save=False
# while True:
#     _, frame1 = cap1.read()
#     _, frame2 = cap2.read()
#     cv2.imshow("image1", frame1)
#     cv2.imshow("image2", frame2)
#     cv2.waitKey(3)

def undistort(img, K, D, DIM, scale=0.6, imshow=False):
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
DIM=(640, 480)
K=np.array([[307.1729192150086, 0.0, 299.38599796182615], [0.0, 307.0689256028552, 240.69982030044062], [0.0, 0.0, 1.0]])
D=np.array([[-0.09728781057723622], [-0.2474835277185543], [0.8093032369811374], [-0.7462057575048991]])
while True:
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    img = undistort(frame2,K,D,DIM)
    cv2.imshow("image1",frame1)
    cv2.imshow("image2",img)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break;
    if im_save==False:
        cv2.imwrite('test.jpg',img)
        im_save=True

cv2.destroyAllWindows()
cap1.release()
cap2.release()