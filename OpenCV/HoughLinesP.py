# Haugh 변환

import cv2
import numpy as np

src = cv2.imread('./data_0419/lec14_building.jpg', cv2.IMREAD_GRAYSCALE)
edge = cv2.Canny(src, 50,150)

lines = cv2.HoughLinesP(edge, 1.0, np.pi / 180. , 160, minLineLength=50, maxLineGap=5)

dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
print(lines.shape)

if lines is not None:
    for i in range(lines.shape[0]):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])
        cv2.line(dst, pt1 , pt2, (0,0,255),2,cv2.LINE_AA)

cv2.imshow("edge", edge)
cv2.imshow('hough', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


'''
# 차선검출 실습 

import cv2
import numpy as np

src = cv2.imread('./data_0419/lec14_lanes.jpg',cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 50,150)

lines = cv2.HoughLinesP(edge, 1.0, np.pi / 180. , 75, minLineLength=50, maxLineGap=5)

print(lines.shape)

if lines is not None:
    for i in range(lines.shape[0]):
        pt1 = (lines[i][0][0], lines[i][0][1])
        pt2 = (lines[i][0][2], lines[i][0][3])
        cv2.line(src, pt1 , pt2, (255,0,0) ,2 ,cv2.LINE_AA)

cv2.imshow("edge", edge)
cv2.imshow('hough', src)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
