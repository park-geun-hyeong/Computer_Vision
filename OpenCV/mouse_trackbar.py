'''
mouse & track bar control
'''

import numpy as np
import cv2

def onChange(value):
    global img, title
    img[:] = value
    cv2.imshow(title, img)


def onMouse(event,x,y,flags, param):
    global img, track_bar, title
    if event == cv2.EVENT_LBUTTONDOWN:
        if [img][0][0] < 246:
            img += 10
    if event == cv2.EVENT_RBUTTONDOWN:
        if [img][0][0] > 10:
            img -= 10

    cv2.setTrackbarPos(track_bar, title, img[0][0])
    cv2.imshow(title, img)


title = 'window'
track_bar = 'track bar'

img = np.zeros((300, 500), np.uint8)
cv2.namedWindow(title)
cv2.imshow(title, img)

cv2.createTrackbar(track_bar, title, 0, 255, onChange)
cv2.setMouseCallback(title, onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()
