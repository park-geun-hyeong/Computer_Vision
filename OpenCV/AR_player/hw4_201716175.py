'''
영상처리 HW4: AR 비디오 플레이어
전자공학부 201716175 박근형
'''

import sys
import numpy as np
import cv2
import time
import os

class HW4:
    def __init__(self, target_fps, target_width, target_height, target_vid_time, output_path):
        '''
        :param target_fps: output video fps(25)
        :param target_width: output video frame width(1280)
        :param target_height: output video frame height(720)
        :param target_vid_time: output video processing time(10)
        :param output_path: output video saving path
        '''

        self.target_fps = target_fps
        self.target_width = target_width
        self.target_height = target_height
        self.output_path = output_path
        self.target_vid_time = target_vid_time
        self.target_frame_num = int(self.target_fps * self.target_vid_time) # 총 프레임 개수 설정
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX') # fourcc 설정
        self.detector = cv2.AKAZE_create() # AkAZE detector 설정
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING) # 특징점 매칭을 위한 BF matcher 설정
        self.delay = int(1000/self.target_fps) # delay 설정

        # output 폴더 만들어주기 
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            print("mkdir output vid path\n")
        
        # 이미지, VideoCapture, VideoWriter 객체 정의해주기
        try:
            self.src_img = cv2.imread("./lec20_korea.jpg", cv2.IMREAD_GRAYSCALE)
            self.cap1 = cv2.VideoCapture('./camvid.avi')
            self.cap2 = cv2.VideoCapture("./lec20_korea.mp4")
            self.out = cv2.VideoWriter(self.output_path + '/output_201716175.avi', self.fourcc, self.target_fps, (self.target_width, self.target_height))

            self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            if self.src_img is None or not self.cap1.isOpened() or not self.cap2.isOpened() or not self.out.isOpened():
                print("load & open failed!")
                sys.exit()
        except FileExistsError:
            print("check your data file!")
            sys.exit()

    def get_homograpy(self, kp1, kp2, desc1, desc2):
        '''
        :param kp1: src_img's keypoint
        :param kp2: frame1's keypoint
        :param desc1: src_img's descriptor
        :param desc2: frame1's descriptor
        :return: homography matrix, homography mask
        '''
        
        # 거리가 가까운 상위 50개의 매칭 설정
        matches = self.matcher.match(desc1, desc2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:50]

        # 호모그래피 계산
        src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
        dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
        mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

        return mat, mask

    def image_composition(self, draw_img, frame2, mat):
        '''
        :param draw_img: draw_img(frame1.copy)
        :param frame2: korea.mp4 frame
        :param mat: homography matrix
        :return: draw_img after image_compositing
        '''

        # korea.mp4프레임 투영변환
        h, w = draw_img.shape[:2]
        dst = cv2.warpPerspective(frame2, mat, (w, h))

        # 합성하기 위한 ROI 마스크 생성
        ROI = np.full(frame2.shape[:2], 255, np.uint8)
        ROI = cv2.warpPerspective(ROI, mat, (w, h))

        # 비디오 프레임을 카메라 프레임에 합성
        cv2.copyTo(dst, ROI, draw_img)
        return draw_img

    def main(self):

        # src 이미지에서 특징점 검출 및 기술자 생성
        kp1, desc1 = self.detector.detectAndCompute(self.src_img, None)
        start = time.time()
        frame_num = 0
        while True:
            ret1, frame1 = self.cap1.read()

            draw_img = frame1.copy()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            if not ret1:
                break

            # 카메라 프레임마다 특징점 검출 및 기술자 생성
            kp2, desc2 = self.detector.detectAndCompute(frame1, None)

            # 특징점이 100개 이상 검출될 경우 매칭 수행
            if len(kp2) > 100:
                
                # 호모그래피 행렬과 바이너리 형태의 매칭 마스크 생성
                mat, mask = self.get_homograpy(kp1, kp2, desc1, desc2)

                # RANSAC 방법에서 정상적으로 매칭된 것의 개수가 20개 이상이면
                if np.sum(mask) >= 20:
                    ret2, frame2 = self.cap2.read()
                    if not ret2:
                        break

                    # 비디오 프레임을 합성하여 AR 프레임 생성
                    draw_img = self.image_composition(draw_img, frame2, mat)

            # AR 프레임 시각화 및 영상으로 저장
            cv2.imshow("frame",  draw_img)
            if cv2.waitKey(self.delay) == 27:
                break

            # AR 프레임 시각화 및 영상으로 저장
            self.out.write(draw_img)

            # 10초의 영상이 모두 생성될 경우 종료
            frame_num += 1
            if frame_num == self.target_frame_num:
                break

        print(f"processing time : {time.time()-start:.4f} sec")
        self.cap1.release()
        self.cap2.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    target_fps = 25
    target_width = 1280
    target_height = 720
    target_vid_time = 10
    output_path = './HW4_result'

    HW = HW4(target_fps, target_width, target_height, target_vid_time, output_path)
    HW.main()
