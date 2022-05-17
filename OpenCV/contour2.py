import cv2
import numpy as np

def setLabel(img, pts, string):
    x,y,w,h = cv2.boundingRect(pts)
    pt1 = (x,y)
    pt2 = (x+w, y+h)
    txt_pt = (x, y-5)
    cv2.rectangle(img, pt1, pt2, (0,0,255), 2)
    cv2.putText(img, string, txt_pt, cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))

if __name__ == "__main__":

    path = "./data_0517/"
    src = cv2.imread(path+'lec17_polygon.bmp', cv2.IMREAD_COLOR)
    cv2.imshow("src", src)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    img = src.copy()

    # 1. grayscale image 이진화
    _ , img_bin = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 배경이 객체보다 intensity가 크기 때문에, THRESH_BINARY_INV 옵션을 사용해 주어 객체정보를 white로 설정하기 위함
    cv2.imshow('binary', img_bin)

    # 2. contour detection
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filtering & Visualization
    for pts in contours:
        if cv2.contourArea(pts) < 400:
            continue

        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02, True)
        vertex = len(approx)

        if vertex == 3:
            setLabel(img, pts, "TRI")
        elif vertex == 4:
            setLabel(img, pts, "RECT")
        else:
            length = cv2.arcLength(pts, True)
            area = cv2.contourArea(pts)
            ratio = 4. * 3.14 * area/ (length ** 2)
            if ratio > 0.85:
                setLabel(img, pts, "Circle")


    cv2.imshow("dst", img)
    cv2.imwrite(path+'output_img.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

