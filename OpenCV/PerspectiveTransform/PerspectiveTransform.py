import cv2
import numpy as np
import sys
import os

def global_th(pos):
    global gray

    _, global_binary = cv2.threshold(gray, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow('global binary', global_binary)

def locally_th(pos):
    global gray

    bsize = pos
    if bsize%2 ==0:
        bsize-= 1
    if bsize<3:
        bsize = 3

    locally_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bsize, 5)
    cv2.imshow("locally binary", locally_binary)

if __name__ == "__main__":
    img_path = "../data_0524/lec18_namecard.jpg"
    try:
        src = cv2.imread(img_path)
        draw_img = src.copy()
    except FileExistsError:
        print("Img not exists")
        sys.exit()

    # cv2.imshow("src", src)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray scale image", gray)

    use_track_bar_for_global = False
    use_track_bar_for_locally = False
    if use_track_bar_for_global:
        cv2.namedWindow('global binary')
        cv2.createTrackbar('threshold', "global binary", 0, 255, global_th)
        cv2.setTrackbarPos('threshold', "global binary", 128)

    if use_track_bar_for_locally:
        cv2.namedWindow("locally binary")
        cv2.createTrackbar('b_size', "locally binary", 0, 200, locally_th)
        cv2.setTrackbarPos('b_size', "locally binary", 11)

    _, img_bin = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary image", img_bin)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for pts in contours:
        if cv2.contourArea(pts) < 400:
            continue

        approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)
        vertex = len(approx)

        if vertex == 4:
            src_pts = approx.reshape(4,2).astype(np.float32)
            cv2.polylines(draw_img, [approx], True, (0,255,0), 2)

    cv2.imshow("draw_img", draw_img)
    width = 720
    height = 480

    target_pts = np.array([[width-1,0], [0,0], [0,height-1], [width-1, height-1]]).astype(np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, target_pts)
    perspective_img = cv2.warpPerspective(src, perspective_matrix, (width, height))
    cv2.imshow("perspective img", perspective_img)
    if perspective_img is not None:
        if not os.path.exists("./HW3_result"):
            os.mkdir("./HW3_result")

        print("Perspectivce_matrix")
        for row in perspective_matrix:
            print(row)
        print("\n")

        np.save("./HW3_result/output.npy", perspective_matrix)
        print("Perspective Matrix saving complete")

        cv2.imwrite("./HW3_result/output.png", perspective_img)
        print("Perspective Image saving complete")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

