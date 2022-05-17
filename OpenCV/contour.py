import cv2
import random
import numpy as np

'''
contour detection
'''

if __name__ == "__main__":

    path = "./data_0517/"
    src = cv2.imread(path+'lec17_contours.bmp', cv2.IMREAD_GRAYSCALE)

    contours,hier = cv2.findContours(src, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    idx = 0
    while idx>=0:
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        cv2.drawContours(dst, contours, idx, color, 2, cv2.LINE_8, hier)
        idx = hier[0, idx, 0]

    cv2.imshow("src",src)
    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

