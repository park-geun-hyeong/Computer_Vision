'''
동영상 처리 
영상 시각화 & 영상 저장

'''
import cv2
import sys
import os

if __name__ == "__main__":
    #  cap = cv2.VideoCapture(0) 실시간 캠으로 받아보기
    path = 'C:/Users/park1/Desktop/2022-1학기/영상처리/실습자료/0322실습자료/'
    cap = cv2.VideoCapture(path + 'lec6_video1.mp4')
    if not cap.isOpened():
        print("check video")
        sys.exit()

    fps = round(cap.get(cv2.CAP_PROP_FPS))
    frame_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    delay = round(1./ fps * 1000)
    print(f"FPS: {fps} image width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, image height:{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, video frame num:{cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    if not os.path.exists(path + 'result/'):
        os.mkdir(path + 'result/')
    out = cv2.VideoWriter(path + 'result/lec6_result_video.avi', fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("videowriter instance error")
        cap.release()
        sys.exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inversed = ~ frame
        out.write(inversed)
        cv2.imshow('frame', frame)
        cv2.imshow('inversed', inversed)

        if cv2.waitKey(delay) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindoes()
