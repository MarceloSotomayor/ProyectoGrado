import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import math

def draw_good_matches(img11, kp11, img21, kp21, matches):
    rows1, cols1 = img11.shape[:2]
    rows2, cols2 = img21.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = np.dstack([img11, img11, img11])
    out[:rows2, cols1:cols1+cols2, :] = np.dstack([img21, img21, img21])

    radius = 4
    BLUE = (255, 0, 0)
    thickness = 1

    for m in matches:
        c1, r1 = kp1[m.queryIdx].pt
        c2, r2 = kp2[m.trainIdx].pt

        cv2.circle(out, (int(c1), int(r1)), radius, BLUE, thickness)
        cv2.circle(out, (int(c2)+cols1, int(r2)), radius, BLUE, thickness)

        cv2.line(out, (int(c1), int(r1)), (int(c2)+cols1, int(r2)), BLUE,
                 thickness)
    return out


MIN_MATCH_COUNT = 10

img2 = cv2.imread('completo.png',0)  # q
img1 = cv2.imread('carta_fin.png',0) # t


sift = cv2.SIFT()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
      
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
    x1 = [np.int32(dst)][0][0][0][0]
    x2 = [np.int32(dst)][0][3][0][0]

    y1 = [np.int32(dst)][0][0][0][1]
    y2 = [np.int32(dst)][0][1][0][1]

    Cx = (x1+x2)/2
    Cy = (y1+y2)/2

    cv2.polylines(img2,[np.int32(dst)],True,(255,0,0),3)
    cv2.circle(img2,(Cx,Cy),7, (255, 255, 255), -1)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = None,
                    matchesMask = matchesMask,
                    flags = 2)

img3 = draw_good_matches(img1,kp1,img2,kp2,good)

print Cx,Cy


#cv2.imshow('SIFT',img2)
plt.imshow(img2)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


