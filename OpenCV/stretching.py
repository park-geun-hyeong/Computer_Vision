'''
image stretching
'''

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
path = './data_0405/'

if __name__ == "__main__":
    src = cv2.imread(path + 'lec10_Hawkes.jpg', cv2.IMREAD_GRAYSCALE)
    dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)
    hist_src = cv2.calcHist([src], [0], None, [256], [0,256])
    hist_dst = cv2.calcHist([dst], [0], None, [256], [0, 256])
    plt.plot(hist_src, 'r', label = 'src')
    plt.plot(hist_dst, 'b', label = 'dst')
    plt.legend()
    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    plt.show()
