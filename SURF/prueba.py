import numpy as np
import cv2
import matplotlib as plt

img1 = cv2.imread('completo.png',0)  # queryImage
img2 = cv2.imread('carta_fin.png',0) # trainImage

rows1, cols1 = img1.shape[:2]
rows2, cols2 = img2.shape[:2]

print rows1,cols1
print rows2,cols2

cv2.imshow('hola',img1)
cv2.imshow('hola1',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
