import cv2
import numpy as np
import sys
from time import time

def KeyMatching(img1, img2,key="AKAZE",good_match_rate=0.11, min_match=10):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#feature-homography

    # [1] ORBを用いて特徴量を検出する
    # Initiate ORB detector
    detector = cv2.AKAZE_create()
    if key=="ORB":
        detector = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),None
    )
    t1 = time()
    kp2, des2 = detector.detectAndCompute(
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),None
    )
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # [2] 検出した特徴量の比較をしてマッチングをする
    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * good_match_rate)]

    # [3] 十分な特徴量が集まったらそれを使って入力画像を変形する
    if len(good) > min_match:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Find homography
        h, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
        t2 = time()
    

        cv2.imwrite('draw_match.jpg', cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2))

        # Use homography
        height, width, channels = img1.shape
        dst_img = cv2.warpPerspective(img2, h, (width, height))
        return dst_img, h,(t2-t1)
    else:
        return img1, np.zeros((3, 3))

file_ref = sys.argv[1]
file_tar = sys.argv[2]
key = sys.argv[3]
img_ref = cv2.imread(file_ref)
img_tar = cv2.imread(file_tar)
#img1 = img_ref[]
#img2 = img_tar[]
img, h,T = KeyMatching(img_ref,img_tar,key)
print("変換行列: \n{}\n処理時間: {}".format(h,T))
#dst = cv2.resize(img, dsize=None, fx=0.1, fy=0.1)

"""
cv2.imshow('test',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""