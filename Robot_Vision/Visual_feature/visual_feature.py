import numpy as np
import cv2
import sys
import time

    
############################ Visual Feature ###################################### 
def visual_feature(kind:str):
    assert type(kind) == str
    
    if kind == 'SIFT':
        al = cv2.xfeatures2d.SIFT_create()
        state = 1
    elif kind == 'SURF':
        al = cv2.xfeatures2d.SURF_create()
        state = 1
    elif kind == 'ORB':
        al = cv2.ORB_create()
        state = 0
    elif kind == 'KAZE':
        al = cv2.KAZE_create()
        state = 1
    else:
        raise Exception("Check Spelling")
    
    start = time.time()
    
    keypoints1, desc1 = al.detectAndCompute(src1, None)
    keypoints2, desc2 = al.detectAndCompute(src2, None)    
    
    print("{} processing time: {:.5f}".format(kind, time.time() - start))
    
    return state, keypoints1, keypoints2, desc1, desc2
    
    #return lists of DMatch object
    #DMatch.distance : 디스크립터 사이의 거리
    #DMatch.trainIdx : train 디스크립터 속의 디스크립터의 인덱스
    #DMatch.queryIdx : query 디스크립터 속의 디스크립터의 인덱스
    #DMatch.imgIdx : train 이미지의 인덱스

############################ BF_Match ######################################
def bf_match(visualfeature, method=None):
    
    state, keypoints1, keypoints2, desc1, desc2 = visualfeature
    
    if state==1: # SIFT, SURF, KAZE
        matcher = cv2.BFMatcher(cv2.NORM_L2) # L2 Norm 사용
    else: # ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # Hamming distance 사용
    
    # 매칭결과를 거리기준 오름차순으로 정렬하여 상위 50개 추출
    BF_match = sorted(matcher.match(desc1, desc2), key = lambda x: x.distance)[:100] 
    
    # BF_match의 queryIdx를 사용하여 첫번째 이미지의 매칭 좌표 구하기
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in BF_match]).reshape(-1,1,2).astype(np.float32)
    
    # BF_match의 trainIdx를 사용하여 두번째 이미지의 매칭 좌표 구하기
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in BF_match]).reshape(-1,1,2).astype(np.float32)
    
    # compare Approximation calculation algorithm
    if method == None:
        # method 따로 없을 경우 상위 50개의 결과 추출
        dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, BF_match, None, flags=2) 
        
    else: 
        # matchmasking을 위한 matrix 추출(essential matrix는 따로 사용하지 않는다.)
        _, M = cv2.findEssentialMat(pts1, pts2, intrinsic_matrix, method = method)
        matchMask = M.ravel().tolist()
        
        # 각각의 mathod에 따른 matchingmask를 설정해 주어 도출된 결과 출력
        dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, BF_match, None, matchesMask = matchMask, flags=2)
         
    return dst, pts1, pts2

############################ BF_knn ######################################
def bf_knn(visualfeature):
    
    state, keypoints1, keypoints2, desc1, desc2 = visualfeature
    
    if state==1: # SIFT, SURF, KAZE
        matcher = cv2.BFMatcher(cv2.NORM_L2) # L2 Norm 사용
    else: # ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # Hamming distance 사용
    
    BF_knn = matcher.knnMatch(desc1, desc2, k = 2) # 매칭할 근접 이웃 개수 2로 설정
    
    good = []
    pts1 = []
    pts2 = []
    for m,n in BF_knn:
        # 두개의 매칭값의 거리를 비교해서 ratio(0.5)안에 포함될 경우 좋은 매칭값이라 판단하고 append
        if m.distance < 0.7*n.distance:          
            good.append([m])         
            #마찬가지로 queryIdx와 trainIdx를 사용하여 매칭포인트 저장해놓기
            pts1.append(keypoints1[m.queryIdx].pt)
            pts2.append(keypoints2[m.trainIdx].pt)
 
    pts1 = np.array(pts1).reshape(-1,1,2).astype(np.float32)
    pts2 = np.array(pts2).reshape(-1,1,2).astype(np.float32)
    
    # 단순 match와는 구별하여 drawMatchesKnn 함수를 사용해주기
    dst = cv2.drawMatchesKnn(src1, keypoints1, src2, keypoints2, good, None, flags=2)

    return dst, pts1, pts2    


############################ flann_match ######################################
def flann_match(visualfeature, method=None):
    
    state, keypoints1, keypoints2, desc1, desc2 = visualfeature
    # 모든 디스크립터를 전수 조사하는 BFmatcher와는 달리 이웃하는 디스크립터를 비교하는 Flann matcher
    
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    
    # flannMatcher 설정을 위한 parameter 정의해주기
    if state == 1: ## SIFT, SURF, KAZE
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    else: # ORB
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number =6, key_size=12, multi_probe_level = 1)
    
    search_params = dict(checks=50) # check: 검색할 후보 수
    
    #위에서 설정한 parameter(index_param, search_param)을 이용하여 flann matcher 정의
    flann = cv2.FlannBasedMatcher(index_params,search_params)   
    
    # 매칭결과를 거리기준 오름차순으로 정렬하여 상위 50개 추출
    flann_match = sorted(flann.match(desc1, desc2), key = lambda x: x.distance)[:50]  

    #마찬가지로 matching keypoints들 정의해주기
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in flann_match]).reshape(-1,1,2).astype(np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in flann_match]).reshape(-1,1,2).astype(np.float32)
    
    # compare Approximation calculation algorithm
    if method == None:
        dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, flann_match, None, flags=2) 
        
    else: 
        _, M = cv2.findEssentialMat(pts1, pts2, intrinsic_matrix, method = method)
        matchMask = M.ravel().tolist()
        dst = cv2.drawMatches(src1, keypoints1, src2, keypoints2, flann_match, None, matchesMask = matchMask, flags=2)
    
    return dst, pts1, pts2  


############################ flann_knn ######################################
def flann_knn(visualfeature):
    
    state, keypoints1, keypoints2, desc1, desc2 = visualfeature
    
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    
    # flannMatcher 설정을 위한 parameter 정의해주기
    if state == 1: ## SIFT, SURF, KAZE
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    else: # ORB
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number =6, key_size=12, multi_probe_level = 1)
   
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)   
    flann_knn = flann.knnMatch(desc1,desc2,k=2) # 매칭할 근접 이웃 개수 2로 설정
    
    # matching mask를 2중 리스트 형식으로 정의
    matchesMask = [[0,0] for i in range(len(flann_knn))]
    pts1=[]
    pts2=[]
    
    
    for i,k in enumerate(flann_knn):
        if len(k) == 2: # 매칭이웃의 개수가 2개모두 나올경우 
            
            # 두개의 매칭값의 거리를 비교해서 ratio(0.5)안에 포함될 경우 좋은 매칭값이라 판단하고 matching mask 설정
            if k[0].distance < 0.7*k[1].distance:
                matchesMask[i]=[1,0]
                pts1.append(keypoints1[k[0].queryIdx].pt)
                pts2.append(keypoints2[k[1].trainIdx].pt)

    # drawMatchesKnn의 keyword args로 사용하기 위하여 다음과 같은 dictionary 정의
    draw_params = dict(matchColor = None,
                       singlePointColor = None,
                       matchesMask = matchesMask,
                       flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)    
           
    pts1 = np.array(pts1).reshape(-1,1,2).astype(np.float32)
    pts2 = np.array(pts2).reshape(-1,1,2).astype(np.float32)
    
    # 위에서 정의한 draw_param을 다음과같이 keyword args로 설정
    dst = cv2.drawMatchesKnn(src1, keypoints1, src2, keypoints2, flann_knn, None, **draw_params)
    
    return dst, pts1, pts2

############################ compare visual feature extracting ######################################   
def compare_visualfeature(SIFT, SURF, ORB, KAZE):
   
    dst1, _, _ = SIFT
    dst2, _, _ = SURF
    dst3, _, _ = ORB
    dst4, _, _ = KAZE
    
    cv2.imshow('SIFT', dst1)
    cv2.imshow('SURF', dst2)
    cv2.imshow('ORB', dst3)
    cv2.imshow('KAZE', dst4)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    

############################ compare matching algorithm ######################################
def compare_matching(BF_match, BF_knn, Flann_match, Flann_knn):
    
    dst1, _, _ = BF_match
    dst2, _, _ = BF_knn
    dst3, _, _ = Flann_match
    dst4, _, _ = Flann_knn
    
    cv2.imshow('BF_match', dst1)
    cv2.imshow('BF_knn', dst2)
    cv2.imshow('Flann_match', dst3)
    cv2.imshow('Flann_knn', dst4)
    
    cv2.waitKey()
    cv2.destroyAllWindows()


############################ compare Approximation calculation algorithm ######################################
def compare_method():
    method = [None, cv2.RANSAC, cv2.LMEDS]
    
    dst=[]
    for m in method:
        d, _, _ = bf_match(visual_feature("SIFT"), method = m)
        dst.append(d)
        
    cv2.imshow('method = None', dst[0])    
    cv2.imshow('method = RANSAC', dst[1])
    cv2.imshow('method = LMEDS', dst[2])
    
    cv2.waitKey()
    cv2.destroyAllWindows()
                
############################ Calculate Essential Matrix ######################################     
def EssentialMatrix(matching):
    _, pts1, pts2 = matching
    
    e, _ = cv2.findEssentialMat(pts1, pts2, intrinsic_matrix)
    print("Essential Matrix :\n {}". format(e))    


if __name__ == "__main__":
    
    '''
    python==3.6.0
    opencv-contrib-python==3.3.0.10
    opencv-python==3.3.0.10
    
    '''
    intrinsic_matrix = np.array([  [506.49469633, 0., 236.84841786],\
                                   [0., 675.16797865, 300.61685296],\
                                   [0., 0., 1.]]).astype(np.float32)
             
    src1 = cv2.resize(cv2.imread('school1.jpg', cv2.IMREAD_GRAYSCALE), dsize=(480,480), interpolation = cv2.INTER_AREA)
    src2 = cv2.resize(cv2.imread('school2.jpg', cv2.IMREAD_GRAYSCALE), dsize=(480,480), interpolation = cv2.INTER_AREA)  

   
    if src1 is None or src2 is None:
        print("Image load failed!")
        sys.exit()
    
    compare_visualfeature(bf_match(visual_feature("SIFT")), bf_match(visual_feature("SURF")), \
                              bf_match(visual_feature("ORB")), bf_match(visual_feature("KAZE")))
    
    compare_matching(bf_match(visual_feature("SIFT")), bf_knn(visual_feature("SIFT")), \
                   flann_match(visual_feature("SIFT")), flann_knn(visual_feature("SIFT"))) 
    
    compare_method()   
    
    EssentialMatrix(flann_match(visual_feature("ORB")))
    
