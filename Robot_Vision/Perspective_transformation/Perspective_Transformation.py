import sys
import numpy as np
import cv2


def drawROI(img, corners):

    '''
    이미지의 모서리를 입력해줄 경우 모서리 포인트와 그들을 이어주는 outline 생성

    :param img: numpy img array
    :param corners: img's corner coordinate
    :return: img marked corner points and outline
    '''
    cpy = img.copy()

    c1 = (192, 192, 255)  ## Point's color
    c2 = (128, 128, 255)  ## Line's color

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 25, c1, -1, cv2.LINE_AA)

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

    return disp


def onMouse(event, x, y, flags, param):  # Function of Mouth Callack
    '''
    마우스 동작으로 이미지를 처리하기 위한 함수

    :param event: mouth statement const ( click, movement etc)
    :param x: window's X point
    :param y: window's Y point
    :param flags: mouth statement when events occurs
    :param param: Data param
    :return: None
    '''

    global srcQuad, dragSrc, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN:  ## 왼쪽 버튼 클릭시 실행
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25:
                dragSrc[i] = True  # 드래그 진행 여부 True(버튼을 잡고 끌었을 경우)
                ptOld = (x, y)  # 클릭하는 순간의 좌표로 갱신
                break

    if event == cv2.EVENT_LBUTTONUP:  ## 왼쪽 버튼을 땠을 때 실행
        for i in range(4):
            dragSrc[i] = False  # 버튼을 땠을 경우엔 드래그 진행 여부 false

    if event == cv2.EVENT_MOUSEMOVE:  ## 마우스가 움직일 때 실행
        for i in range(4):
            if dragSrc[i]:  ## 드래가 진행 여부 True 일 경우
                dx = x - ptOld[0]  # 드래그 진행하여 이동한 x 변화량
                dy = y - ptOld[1]  # 드래그 진행하여 이동한 y 변화량

                srcQuad[i] += (dx, dy)  # 기존의 모서리 좌표에서 dx,dy를 더해 드래그 이동 후의 좌표 설정

                cpy = drawROI(src, srcQuad)  # outline 모서리 따라서 이동
                cv2.imshow('img', cpy)
                ptOld = (x, y)  # 드래가 이후의 좌표로 갱신
                break


# 입력 이미지 불러오기
src = cv2.imread('img2.jpg')

if src is None:
    print('Image open failed!')
    sys.exit()

# 입력 영상 크기 및 출력 영상 크기
h, w = src.shape[:2]
dw = 500
dh = round(dw * 297 / 210)  # A4 용지 크기: 210x297cm

# 모서리 점들의 좌표, 드래그 상태 여부
srcQuad = np.array([[30, 30], [30, h - 30], [w - 30, h - 30], [w - 30, 30]], np.float32)  # 처음에 outline이 찍히는 모서리 좌표
dstQuad = np.array([[0, 0], [0, dh - 1], [dw - 1, dh - 1], [dw - 1, 0]], np.float32)  ## sacan할 문서의 최종 outline이 찍히는 모서리 좌표
dragSrc = [False, False, False, False]

# 모서리점, 사각형 그리기
disp = drawROI(src, srcQuad)

cv2.imshow('img', disp)
cv2.setMouseCallback('img', onMouse)

while True:
    key = cv2.waitKey()
    if key == 13:  # ENTER 키
        break
    elif key == 27:  # ESC 키
        cv2.destroyWindow('img')
        sys.exit()

# 투시 변환
pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)  ## Get perspective transform matrix
dst = cv2.warpPerspective(src, pers, (dw, dh),
                          flags=cv2.INTER_CUBIC)  # Get changed image using perspective transform matrix

print(pers)
print(pers.shape)

# 결과 영상 출력
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()