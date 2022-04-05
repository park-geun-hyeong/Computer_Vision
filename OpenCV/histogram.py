'''
image histogram
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
path = './data_0405/'

if __name__ == "__main__":
    img_path = path + 'lec10_lenna.bmp'
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    dst = np.clip(src.astype(np.int32) + 50, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1,2, figsize=(12,4))

    hist_src = cv2.calcHist([src], [0], None, [256], [0,256])
    hist_dst = cv2.calcHist([dst], [0], None, [256], [0,256])
    ax[0].plot(hist_src, 'r', label = 'src')
    ax[0].plot(hist_dst, 'b', label = 'dst')
    ax[0].legend()
    ax[0].set_title("gray scale")

    color = ['b', 'g', 'r']
    color_src = cv2.imread(img_path)
    bgr = cv2.split(color_src)

    for (p, c) in zip(bgr, color):
        hist = cv2.calcHist([p], [0], None, [256], [0, 256])
        print(hist.shape)
        ax[1].plot(hist, color = c, label = c)

    ax[1].legend()
    ax[1].set_title("color scale")
    #cv2.imshow('img',img)
    plt.show()
