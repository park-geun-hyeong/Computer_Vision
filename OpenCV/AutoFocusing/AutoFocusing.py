import cv2
import sys
import os
import numpy as np

def cal_sharpness(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize = 5)

    #case 1
    # height, width = gray.shape
    # pixel_num = height * width
    # pixel_mean = np.mean(np.abs(laplacian))
    # stddev1 = np.sqrt(np.sum([(i-pixel_mean)**2 for i in np.abs(laplacian).ravel()])/pixel_num)

    # case 2
    # stddev2 = cv2.meanStdDev(np.abs(laplacian))[1][0][0]

    #case 3
    stddev = np.std(np.abs(laplacian.ravel()))

    return laplacian, stddev

if __name__ == "__main__":

    vid_path = "./lec13_autofocus.mp4"
    result_path = "./result/"
    result_img = result_path + "HW2_201716175.png"

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Video open error")
        sys.exit()

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    delay = round(1000 / fps)
    print(f"FPS: {fps} image width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, image height:{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, video frame num:{cap.get(cv2.CAP_PROP_FRAME_COUNT)}, video time: {frame_num/fps:.2f} sec")

    imgs = []
    MAX_sharpness = -1
    MAX_FRAME_NUM = 0
    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        laplacian, sharpness = cal_sharpness(frame)
        if sharpness > MAX_sharpness:
            MAX_sharpness = sharpness
            MAX_FRAME_NUM = cnt

        TXT = f"Sharpness: {sharpness:.4f}"
        cv2.rectangle(frame, (frame_width - 400, frame_height - 60), (frame_width - 50, frame_height - 10), (255,255,255), -1)
        cv2.putText(frame, TXT, (frame_width - 350, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        imgs.append(frame)
        cnt += 1
        cv2.imshow("frame", frame)
        cv2.imshow("laplacian", laplacian)
        cv2.waitKey(delay)

    cap.release()
    cv2.destroyAllWindows()

    ANS_IMG = imgs[MAX_FRAME_NUM]
    cv2.imwrite(result_img, ANS_IMG)
    print("imwrite complete")

    cv2.imshow(f"Most Sharpness Frame({MAX_FRAME_NUM}th frame)", ANS_IMG)
    cv2.waitKey(0)











