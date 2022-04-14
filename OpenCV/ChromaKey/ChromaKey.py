import cv2
import sys
import os

def show_video_info(cap):
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    return fps, width, height, frame_num

if __name__ == "__main__":

    cap1 = cv2.VideoCapture('./lec7_raining.mp4')
    cap2 = cv2.VideoCapture('./lec7_woman.mp4')

    if not cap1.isOpened() or not cap2.isOpened():
        print("check videos are exits")
        sys.exit()

    fps1, frame_width1, frame_height1, frame_num1 = show_video_info(cap1)
    fps2, frame_width2, frame_height2, frame_num2 = show_video_info(cap2)
    print(f"Video1 =>> FPS: {fps1} image width: {frame_width1}, image height:{frame_height1}, video frame num:{frame_num1}")
    print(f"Video2 =>> FPS: {fps2} image width: {frame_width2}, image height:{frame_height2}, video frame num:{frame_num2}")

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    delay = round(1./ fps1 * 1000)

    if not os.path.exists('./result/'):
        os.mkdir('./result/')

    out = cv2.VideoWriter('./result/hw1_201716175.avi', fourcc, fps1, (frame_width1, frame_height1))
    if not out.isOpened():
        print("videowriter error")
        cap1.release()
        cap2.release()
        sys.exit()

    space = True
    cnt = 0
    while True:
        ret1, woman_frame = cap2.read()

        if not ret1:
            break

        if space:
            ret2, raining_frame = cap1.read()

            if not ret2:
                break

            hsv = cv2.cvtColor(woman_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (50, 150, 0), (70, 255, 255))
            cv2.imshow('mask',mask)
            cv2.copyTo(raining_frame, mask, woman_frame)

        cnt += 1
        txt1 = "201716175/Park-Geun-Hyeong"
        txt2 = f"Space: {space}, Frame_Num: {cnt}"
        cv2.putText(woman_frame, txt1, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(woman_frame, txt2, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow('frame', woman_frame)
        out.write(woman_frame)
        key = cv2.waitKey(15)

        if key == ord(' '):
            space = not space
        elif key == 27:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
