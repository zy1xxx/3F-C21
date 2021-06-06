import cv2
import numpy as np

def CropImage(img):
        sp = img.shape  
        sz1 = sp[0]  
        sz2 = sp[1]  
        a = int(sz1 / 2 - 100)  # x start
        b = int(sz1 / 2 + 100)  # x end
        c = int(sz2 / 2 - 150)  # y start
        d = int(sz2 / 2 + 150)  # y end
        cropImg = img[a:b, c:d]  
        return cropImg

def Distance(img):
    sp=img.shape
    countPoint=0
    for j in range(0, sp[1]):
        for i in range(0, sp[0]):
            if img[i][j]==255 :
                countPoint+=1
    percentage=float(countPoint/(sp[0]*sp[1]))
    return percentage

def Direction(img):
    left=img[1:200,1:150]
    right = img[1:200,151:300]
    countPointL=0
    countPointR=0
    for j in range(0, 149):
        for i in range(0, 199):
            if left[i][j]==255 :
                countPointL += 1
            if right[i][j]==255:
                countPointR+=1
    if countPointR>countPointL:
        return False
    else:
        return True


def RedMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([0, 43, 46])
    high_hsv = np.array([10, 255, 255])
    img = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    kernel = np.ones((5, 5), np.int)
    eroded = cv2.erode(img, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated

def YellowMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([11, 43, 46])
    high_hsv = np.array([25, 255, 255])
    img = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    kernel = np.ones((10, 10), np.int)
    eroded = cv2.erode(img, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated
def blockDetect(img):
    constant=0.01 #start turn
    stopFlag=False
    cropedImg=CropImage(img)
    red=RedMask(cropedImg)
    yellow=YellowMask(cropedImg)
    cv2.imshow('red', red)
    cv2.imshow('yellow', yellow)
    print(Distance(red))
    if Distance(red)>constant:
        turnRight = Direction(yellow)
        print(turnRight)
        stopFlag=True
        return turnRight,stopFlag
    else :
        return False,stopFlag


def turn(turnRight):
    if turnRight:
        print("turn right") #turn right
    else :
        print("turn left") #turn left
    return False #stopFlag=False


cap2 = cv2.VideoCapture(1)
while True:
    _, image = cap2.read()
    cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
    cv2.imshow("image2",frame2)
    turnRight,stopFlag=blockDetect(image)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break;
    if im_save==False:
        cv2.imwrite('test.jpg',frame2)
        im_save=True
    if stopFlag:
        turn(turnRight)
    else:
        print("run") #normally run
cv2.destroyAllWindows()
cap1.release()
cap2.release()
