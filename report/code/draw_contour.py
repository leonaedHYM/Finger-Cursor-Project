import numpy as np
import cv2


gesture1 = cv2.imread("E:/Northwestern University/Courses/2021fall courses/332_CV/final_Project/gray_han2.png")
gray = cv2.cvtColor(gesture1, cv2.COLOR_BGR2GRAY)
kernel_size = 5
kernel1 = np.ones((kernel_size,kernel_size),np.float32)/kernel_size/kernel_size
kernel2 = np.ones((10,10), np.uint8)/100
#skin_min = np.array([0, 40, 50],np.uint8)  # HSV mask
#skin_max = np.array([30, 250, 255],np.uint8)
#hsv = cv2.cvtColor(gesture1, cv2.COLOR_BGR2HSV)
#mask = cv2.inRange(hsv, skin_min, skin_max)
#res = cv2.bitwise_and(hsv, hsv, mask= mask)
#res = cv2.erode(hsv, kernel1, iterations=1)
#res = cv2.dilate(hsv, kernel1, iterations=2)
#rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gesture1, (11, 11), 0)
blur = cv2.blur(gray, (10,10))
ret, thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(contours, key = cv2.contourArea)
cv2.drawContours(gesture1, [c], -1, (0, 255, 0), 10)
#cv2.namedWindow('Contours',cv2.WINDOW_NORMAL)
cv2.imwrite('contour_han2_biggest.png',gesture1)
        # _, gray = cv2.threshold(gray,30, 255,cv2.THRESH_BINARY)
        # gray = cv2.dilate(gray, kernel2, iterations=1)
        # cv2.imshow('2d',gray)

        ## Canny edge detection at Gray space.
#canny = cv2.Canny(gray, 100, 200)
#output = cv2.erode(res, kernel1, iterations=1)
#output = cv2.dilate(res, kernel2, iterations=1)
#cv2.imwrite('output.png',output)

#canny = cv2.Canny(gesture2, 100, 200)
#output , _ = cv2.findContours(gesture2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.imshow('contours',output[0])
#canny = cv2.erode(output, kernel1, iterations=1)
#canny = cv2.dilate(canny, kernel1, iterations=1)
#cv2.imshow('edges',canny)
cv2.waitKey()
'''

gesture2 , _ = cv2.findContours(gesture2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(gesture1.shape)
print(gesture2[0].shape)
#cv2.waitKey()
'''