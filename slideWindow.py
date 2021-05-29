import cv2
import numpy as np
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
#读取图像
img = cv2.imread("./ImgTest/out.png")
#将原图转为灰度图
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Canny边缘检测
canny_img = cv2.Canny(gray_img,100,150,3)
print(canny_img.shape)
#slide
Pheight=img.shape[0]
Pwidth=img.shape[1]
print(img.shape)
width=200
height=40
linewidthMin=20
originPoint=(int(Pwidth/2-width/2),Pheight-height)
canny_img2=canny_img.copy()
cv2.rectangle(canny_img2, (originPoint[0],originPoint[1]), (originPoint[0]+width, originPoint[1]+height), (255,0,255), 2)
cv2.imshow("canny_img",canny_img)
# for i in range(500):
#     print("point",(0+i,474),canny_img[0+i][474])

for v in range(int(Pheight/height)):
    print("rec index:",v)
    print(originPoint)
    position=0
    positionCnt=0
    for j in range(3):
        cnt = 0
        poSum = 0
        line1=False
        restart=0
        for i in range(width):
            if line1==False:
                try:
                    if canny_img[originPoint[1]+j][originPoint[0]+i] ==255:
                        cnt+=1
                        poSum=poSum+i
                        restart=i+linewidthMin
                        line1=True
                        #print(i)
                except:
                    pass
            else:
                if i<restart:
                    continue
                else:
                    try:
                        if canny_img[originPoint[1]+j][originPoint[0]+i] ==255:
                            cnt += 1
                            poSum = poSum + i
                            #print(i)
                            break
                    except:
                        pass
        if cnt!=0:
            linePo=poSum/cnt
            # print('cnt:',cnt)
            position=position+linePo
            positionCnt+=1
    if positionCnt!=0:
        position=position/positionCnt
    print("aver:",position)
    print("<<<<<<")
    newPoint=(int(originPoint[0]+position-width/2),originPoint[1]-height)
    cv2.rectangle(canny_img2, (newPoint[0],newPoint[1]), (newPoint[0]+width,newPoint[1]+height), (255,0,255), 2)
    originPoint=newPoint
# print(newPoint)
# print((newPoint[0]+width,newPoint[1]+height))
cv2.imshow("canny_img",canny_img2)

cv2.setMouseCallback("canny_img", on_EVENT_LBUTTONDOWN)
cv2.waitKey(0)
