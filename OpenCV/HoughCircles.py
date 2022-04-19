import cv2
import numpy as np

src = cv2.imread('./data_0419/lec14_iris.jpg', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
blr = cv2.GaussianBlur(gray, (0,0), 0.5)

circle = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 10, param1 = 100, param2 = 40, minRadius = 10, maxRadius = 60)
dst = src.copy()
print(circle)

if circle is not None:
    for i in range(circle.shape[1]):
        cx, cy, radius = np.uint16(circle[0][i])
        cv2.circle(dst, (cx,cy), radius, (0,255,0), 2, cv2.LINE_AA)
        cv2.circle(dst, (cx,cy), 5, (0,0,255), -1, cv2.LINE_AA)

cv2.imwrite('./result.png', dst)
cv2.imshow('result',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

