import numpy as np
import cv2
import sys

if __name__ == "__main__":
    path = './data_0531/'
    src1 = cv2.imread(path + 'lec20_book.png', cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread(path + 'lec20_book_in_scene.png', cv2.IMREAD_GRAYSCALE)

    detector = cv2.KAZE_create()

    kp1, desc1 = detector.detectAndCompute(src1, None)
    kp2, desc2 = detector.detectAndCompute(src2, None)

    matcher = cv2.BFMatcher_create()
    matches = matcher.match(desc1, desc2)

    good_matches = sorted(matches, key = lambda x : x.distance)[:50]

    result = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # find homography
    pt1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2).astype(np.float32)
    pt2 = np.array([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2).astype(np.float32)

    homo_mtx, _ = cv2.findHomography(pt2, pt1, cv2.RANSAC)
    for i in homo_mtx:
        print(i)

    h, w = src1.shape[:2]
    homo = cv2.warpPerspective(src2, homo_mtx, (w, h))

    # cv2.imshow('src1 ', src1)
    # cv2.imshow('src2', src2)
    cv2.imshow('matching', result)
    cv2.imshow('homo', homo)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



