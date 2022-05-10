import cv2
import numpy as np

if __name__ == "__main__":

    path = "./data_0510/"
    src = cv2.imread(path+'lec16_rice.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('src', src)

    # adaptiveThreshold 함수를 사용하여 지역이진화 해주기 
    bsize = 181
    C = -15
    binary_img = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, C)
    cnt1, _ = cv2.connectedComponents(binary_img)
    cv2.imshow('binary', binary_img)
    print(f"Nums before Opening: {cnt1}")

    # 직접 구역을 나눠주어 Threshold 함수를 사용하여 지역이진화 해주기
    dst1 = np.zeros(src.shape, np.uint8)
    bw = src.shape[1]
    bh = src.shape[0]

    for y in range(4):
        for x in range(4):
            src_ = src[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
            dst_ = dst1[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
            cv2.threshold(src_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst_)

    cv2.imshow('binary_th', dst1)

    open_dst = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, None)
    cv2.imshow("open", open_dst)
    cnt2, _ = cv2.connectedComponents(open_dst)
    print(f"Nums after Opening: {cnt2}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
