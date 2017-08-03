import cv2
import numpy as np

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




