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
    dst_eq = cv2.equalizeHist(src)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    hist_src = cv2.calcHist([src], [0], None, [256], [0,256])
    hist_dst = cv2.calcHist([dst], [0], None, [256], [0, 256])
    hist_eq = cv2.calcHist([dst_eq], [0], None, [256], [0, 256])
    ax[0].plot(hist_src, 'r', label = 'src')
    ax[0].plot(hist_dst, 'b', label = 'dst')
    ax[0].legend()
    ax[0].set_title('normalize')

    ax[1].plot(hist_src, 'r', label='src')
    ax[1].plot(hist_eq, 'g', label='dst')
    ax[1].set_title('equalization')
    ax[1].legend()

    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    cv2.imshow('dst_eq', dst_eq)
    plt.show()
