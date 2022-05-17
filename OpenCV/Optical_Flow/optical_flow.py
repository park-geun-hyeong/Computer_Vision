import numpy as np
import cv2
import sys


if __name__ == "__main__":
    cap = cv2.VideoCapture("./vtest.avi")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    delay = 1000//fps

    color = np.random.randint(0,255, (200, 3))

    lines = None
    prevImg = None
    termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    while True:
        ret, frame =cap.read()
        if not ret:
            break
        
        img_draw = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prevImg is None:
            prevImg = gray
            lines = np.zeros_like(frame)
            prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)
            
        else:
            nextImg = gray
            nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPt, None, criteria = termcriteria)

            prevMv = prevPt[status == 1]
            nextMv = nextPt[status == 1]
            
            for i,(p,n) in enumerate(zip(prevMv, nextMv)):
                px,py = p.ravel()
                nx,ny = n.ravel()

                cv2.line(lines, (px,py), (nx,ny), color[i].tolist(),2)
                cv2.circle(img_draw, (nx,ny), 2, color[i].tolist(), -1)


            img_draw = cv2.add(img_draw, lines)
            prevImg = nextImg
            prevPt = nextPt.reshape(-1,1,2)

        cv2.imshow("opticalflow-LK", img_draw)
        key = cv2.waitKey(delay)
        if key ==27:
            break
        elif key == 8:
            prevImg = None


    cv2.destroyAllWindows()
    cap.release()


