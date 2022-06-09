import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################################

brushThickness = 10
eraserThickness = 100

#######################################
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
imgCanvas.fill(255)

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

drawColor = (0, 0, 255)

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
shape = 'circle'

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85, maxHands=1)

xp, yp = 0, 0

while True:

    # import Image
    success, img = cap.read()

    img = cv2.flip(img, 1)

    # Find hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #img.fill(255)
    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x0, y0 = lmList[4][1:]

        # check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)

        # if selection mode - Two Fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 110:
                if 250 < x1 < 400:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 420 < x1 < 540:
                    header = overlayList[1]
                    drawColor = (0, 0, 0)
                elif 560 < x1 < 670:
                    header = overlayList[2]
                    drawColor = (255, 0, 255)
                elif 700 < x1 < 800:
                    header = overlayList[3]
                    shape = 'circle'
                elif 820 < x1 < 940:
                    header = overlayList[4]
                    shape = 'rectangle'
                elif 950 < x1 < 1100:
                    header = overlayList[5]
                    shape = 'eclipse'
                elif 1120 < x1 < 1230:
                    header = overlayList[6]
                    drawColor = (255, 255, 255)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (255, 0, 255), cv2.FILLED)


        # if Drawing Mode - if Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1


            if drawColor == (255, 255, 255):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1


            #Rectangle
            if shape == 'rectangle':
                z1, z2 = lmList[4][1:]
                # print(z1,z2)
                result = int(((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                # print(result)
                if result < 0:
                    result = -1 * result
                u = result
                cv2.rectangle(img, (x0, y0), (x1, y1), drawColor)
                cv2.putText(img, "Length of Diagonal = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                cv2.putText(img, str(u), (530, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                if fingers[4]:
                    cv2.rectangle(imgCanvas, (x0, y0), (x1, y1), drawColor)
                    

            # Circle
            if shape == 'circle':
                z1, z2 = lmList[4][1:]
                # print(z1,z2)
                result = int(((((z1 - x1) ** 2) + ((z2 - y1) ** 2)) ** 0.5))
                # print(result)
                if result < 0:
                    result = -1 * result
                u = result
                cv2.putText(img, "Radius Of Circe = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv2.putText(img, str(u), (450, 700), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv2.circle(img, (x0, y0), u, drawColor)
                if fingers[4]:
                    cv2.circle(imgCanvas, (x0, y0), u, drawColor)

            # Ellipse
            if shape == 'eclipse':
                z1, z2 = lmList[4][1:]
                # cv2.ellipse(img,(x1,y1),(int(z1/2),int(z2/2)),0,0,360,255,0)
                a = z1 - x1
                b = (z2 - x2)
                if x1 > 250:
                    b = int(b / 2)
                if a < 0:
                    a = -1 * a
                if b < 0:
                    b = -1 * b
                cv2.ellipse(img, (x1, y1), (a, b), 0, 0, 360, 255, 0)
                cv2.putText(img, "Major AL, Minor AL = ", (0, 700), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                cv2.putText(img, str(a), (550, 700), cv2.FONT_HERSHEY_PLAIN, 2, (123, 20, 255), 2)
                cv2.putText(img, str(b), (700, 700), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                if fingers[4]:
                    cv2.ellipse(imgCanvas, (x1, y1), (a, b), 0, 0, 360, 255, 0)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    #_, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    #imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    #img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the Header Image
    img[0:110, 0:1280] = header
    imgCanvas[0:110, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    #cv2.imshow("WhiteBoard", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
