import sys
import numpy as np
import cv2


def on_mouse(event, x, y, flags, param):  # Function of Mouth Callack
    '''
     마우스 동작으로 이미지를 처리하기 위한 함수

     :param event: mouth statement const ( click, movement etc)
     :param x: Clicked X point on Image
     :param y: Clicked Y point on Image
     :param flags: mouth statement when events occurs
     :param param: Data param
     :return: None
     '''

    global cnt, src_pts # global 변수 설정 : cnt(마우스 클릭 수), src_pts(클릭되는 모서리 좌표를 저장하기 위한 array)
    if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽 버튼 클릭시 실행
        if cnt < 4: # 4개의 모서리를 다 누르기 전까지 실행
            src_pts[cnt, :] = np.array([x, y]).astype(np.float32) # cnt 번째 클릭 시 x,y 좌표는 src_pts 의 cnt 행에 저장된다.
            cnt += 1 # 클릭 이후 cnt 변수에 1 더해주기

            cv2.circle(src, (x, y), 5, (0, 0, 255), -1) # 클릭된 x,y좌표(모서리)를 cv2.circle 함수를 통해 시각화
            cv2.imshow('src', src) # 마우스 동작이 일어나는 동안 src 이미지 계속 보여주기

        if cnt == 4: # 마우스 클릭을 4번째 했을 경우(물체의 모든 모서리를 다 클릭했을 경우)
            
            # dst 이미지 width, height 기준크기 설정해주기
            w = 350
            h = 550

            # Perspective transformation 이 일어난 destination image 의 모서리 좌표 설정해 주기 위함
            # [[0,0], [349,0] ,[349, 549], [0, 549]]
            dst_pts = np.array([[0, 0], # 좌상단 좌표
                                [w - 1, 0],
                                [w - 1, h - 1], # 우하단 좌표
                                [0, h - 1]]).astype(np.float32)  

            pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # src 좌표와 dst 좌표 사이의 perspective transformation matrix 구하기

            dst = cv2.warpPerspective(src, pers_mat, (w, h))
            # perspective transformation 이 적용된 dst 이미지 생성

            cv2.imshow('dst', dst) # dst 이미지 띄어주기


cnt = 0  # 왼쪽 마우스 버튼의 클릭수를 새줄 cnt 변수
src_pts = np.zeros([4, 2], dtype=np.float32)  # src 이미지 속 물체의 모서리 좌표가 들어갈 4행 2열의 array 를 0으로 초기화

src = cv2.imread('img2.jpg') # src 이미지 읽어오기
if src is None: # src 이미지 불러오는 것을 실패했을 경우의 예외처리
    print('Image load failed!')
    sys.exit()

cv2.namedWindow('src')  # 'src' 라는 이름의 window 생성
cv2.setMouseCallback('src', on_mouse) # 위에서 정의한 on_mouse 함수를 적용하여 마우스 동작시키기

cv2.imshow('src', src)  # src 이미지 띄우기
cv2.waitKey(0)  # key 입력이 있을때까지 window 창 무한 대기
cv2.destroyAllWindows()  # 화면에 나타난 window 창 종료
