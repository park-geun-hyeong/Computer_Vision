import numpy as np
import cv2
import sys
import time

realwidth = 1.5  # object's real width(meter)

def distance(focallength, realwidth, pixelwidth):
    '''
    Estimate real distance using camera calibration

    :param focallength: focal_length
    :param realwidth: object's real width
    :param pixelwidth: object's pixel width on frame
    :return: real distance (meter)
    '''

    dist = (realwidth * focallength) / pixelwidth
    return dist

def video(FocalLength, video_path, kind:str):
    '''
    Object Detection & Distance Estimation(using visual feature)
    Show visualized bounding box & distance txt on video

    :param FocalLength: focal_length
    :param video_path: video path
    :param kind: detecting algorithm(str)
    :return: None
    '''

    vidcap = cv2.VideoCapture(video_path)

    # set detect & matching algorithm
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6

    if kind == 'KAZE':
        al = cv2.KAZE_create()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    elif kind == "ORB":
        al = cv2.ORB_create()
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)

    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    if not vidcap.isOpened():
        print('Video open failed!')
        sys.exit()

    cnt = 0
    success = True

    # read video cap
    while success:
        success, image = vidcap.read()
        cnt += 1

        try:
            if cnt == 1: # first frame
                first_frame = image
                sub_img = first_frame[290:490, 85:320] # crop sub image
                base_kp, base_dec = al.detectAndCompute(sub_img, None)

            elif cnt % 10 == 0: # 10*n frame
                kp, dec = al.detectAndCompute(image, None)
                matches = matcher.match(base_dec, dec)

                matches = sorted(matches, key=lambda x: x.distance)[:20]

                src_pts = np.float32([base_kp[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])

                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                h, w = sub_img.shape[:2]
                pts = np.float32([[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]])
                dst = cv2.perspectiveTransform(pts, mtrx)

                cv2.rectangle(image, (int(dst[0][0][0]), int(dst[0][0][1])), (int(dst[2][0][0]), int(dst[2][0][1])), (0, 255, 0), thickness=2)
                try:
                    dist = distance(FocalLength, realwidth, abs(int(dst[0][0][0]) - int(dst[2][0][0])))
                except ZeroDivisionError:
                    continue

                txt = "{:.2f} M".format(dist)
                cv2.putText(image, txt, (int(dst[0][0][0]-1), int(dst[0][0][1])-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                ###############################################################################
                # if you want to see matching video, uncomment this part

                #matchesMask = mask.ravel().tolist()
                #dst = cv2.drawMatches(sub_img, base_kp, image, kp, matches, None, \
                #                       matchesMask = matchesMask,
                #                       flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                #cv2.imshow('matching vid', dst)
                ###############################################################################

                cv2.imshow('vid', image)

            else:
                continue

        except:
            print("cnt: {}, dst: {}".format(cnt, dist))
            continue


        if cv2.waitKey(10) == 27:
            break


if __name__ == "__main__":

    focal_length = 506.49469633

    start = time.time()
    video(focal_length, "./car2.mp4", "KAZE")

    print("processing time : {:.4f} sec".format(time.time() - start))





