import cv2
import numpy as np
'''
labeling on binary image
'''

if __name__ == "__main__":

    path = "./data_0517/"
    src = cv2.imread(path+'lec17_keyboard.bmp', cv2.IMREAD_GRAYSCALE)

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin) # 이진영상 labeling
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    for i in range(1, cnt):
        x,y,w,h,area = stats[i]
        if area<20:
            continue

        cv2.rectangle(dst, (x,y,w,h), (0,255,255))


    cv2.imshow("src",src)
    cv2.imshow("src bin",src_bin)
    cv2.imshow('label', labels.astype(np.uint8))
    cv2.imshow("dst",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

