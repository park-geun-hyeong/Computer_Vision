import numpy as np
import cv2

drawing = False # mouse click 상태확인
mode = True #  true: 사각형, false: 원
ix, iy = -1, -1

def draw_circle(event, x,y, flags, param):
    global ix,iy,drawing,mode, draw_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy= x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(draw_img, (ix,iy), (x,y) ,(0,255,255), 1)
            cv2.imshow('window', draw_img)
            draw_img = img.copy()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img, (ix, iy), (x,y), (0,255,0),2)
        else:
            cv2.circle(img,(x,y),5,(0,0,255), -1)
        cv2.imshow('image', img)


if __name__ =="__main__":
    img = np.zeros((512,512,3), np.uint8)
    draw_img = img.copy()

    cv2.namedWindow('window')
    cv2.setMouseCallback('window', draw_circle)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == ord('a'):
            mode = not mode
        elif k == 27:
            break

    cv2.destroyAllWindows()
