import cv2
import os
import numpy as np
from datetime import datetime
from pathlib2 import Path


DIM_C=(640,480)
K_C=np.array([[264.8718530264331, 0.0, 299.52281262869144],
          [0.0, 264.8614748244235, 241.52849384953572],
          [0.0, 0.0, 1.0]])
D_C=np.array([[-0.018465666357146516],
            [0.00023640864891298283],
            [0.0004638369263624063],
            [-0.0010282773327839974]])

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


cap = cv2.VideoCapture(1)
base_path=os.path.dirname(os.path.abspath(__file__))
view_path=os.path.join(base_path,'block_views/')
if not os.path.exists(view_path):
    os.mkdir(view_path)
print (view_path)

i = 0
while (1):
    _, frame = cap.read()
    frame=undistort(frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('s'):
        picture_path=os.path.join(view_path,str(datetime.now().strftime('%Y-%m-%d_%H-%M'))+str(i).zfill(2)+'.png')
        cv2.imwrite( picture_path, frame)
        i += 1
    cv2.namedWindow("capture",cv2.WINDOW_NORMAL)
    cv2.imshow("capture", frame)

cv2.destroyAllWindows()
cap.release()

