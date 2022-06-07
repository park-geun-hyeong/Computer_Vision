import numpy as np
import cv2
import sys

if __name__ == "__main__":
    path = './vtest.avi'
    cap = cv2.VideoCapture(path + "pedestrians.avi")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    out = cv2.VideoWriter("./outpub.mp4", fourcc, 20, (width, height))

    ret, prev = cap.read()
    hsv = np.zeros_like(prev)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    hsv[..., 1] = 255

    while True:
        ret, now = cap.read()
        if not ret:
            break

        cv2.imshow("src", now)
        now = cv2.cvtColor(now, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, now, None, 0.5, 3, 13, 3, 5, 1.1, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        print(mag.shape, ang.shape)
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("flow", bgr)

        out.write(bgr)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cap.release()
    out.release()


