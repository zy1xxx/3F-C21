import numpy as np
import cv2
import matplotlib.pyplot as plt

#
#利用单应性查找查找对象

MIN_MATCH_COUNT=15

img11 = cv2.imread('F:\MyWork\ETI\IVE\Views/1.png',0)
img12 = cv2.imread('F:\MyWork\ETI\IVE\Views/2.png',0)

img2 = cv2.imread('F:\MyWork\ETI\IVE\Views/00.png',0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img11,None)
kp12, des12 = orb.detectAndCompute(img12,None)
kp2, des2 = orb.detectAndCompute(img2,None)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks = 100)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img11.shape
    print(img2.shape)
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    x_mid=0
    y_mid=0
    for coordinate in dst:
        x_mid+=coordinate[0][0]
        y_mid+=coordinate[0][1]
    x_mid = int(x_mid / dst.shape[0])
    y_mid = int(y_mid / dst.shape[0])
    print(x_mid)
    print(y_mid)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    img2 = cv2.putText(img2, 'Beer', (x_mid, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
else:
    print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # 用绿色绘制匹配
                   singlePointColor = None,
                   matchesMask = matchesMask, # 只绘制内部点
                   flags = 2)

img3 = cv2.drawMatches(img11,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()
