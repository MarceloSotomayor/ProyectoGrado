#Importar las librerias necesarias
import numpy as np
import cv2
import argparse
import time
from matplotlib import pyplot as plt


#Definir funcion para dibujar los nuevos puntos
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

#Parsear los argumentos si es necesario
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",help="Opcional en caso de usar video")
ap.add_argument("-t","--train",required=True,help="Imagen con el objeto a seguir")
args = vars(ap.parse_args())


#Definir si se usa video o camara web, validando algunas situaciones
if not args.get("video",False):
    camara = cv2.VideoCapture(2)
    print "[INFO] Calentando la camara ..."
    time.sleep(2)
    camara.set(3,640)
    camara.set(4,480)
    time.sleep(2)

    if camara.isOpened():
        print "[INFO] Camara encendida"
else:
    camara = cv2.VideoCapture(args["video"])


#Numero minimo de coincidencias requeridas para detectar al objeto
MIN_MATCH_COUNT = 10

#Imagen de entrenamiento 
img1 = cv2.imread(args["train"],0) # t

#Inicializar algoritmo SIFT
sift = cv2.SIFT()

kp1, des1 = sift.detectAndCompute(img1,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    
    (grabbed, img2_rgb) = camara.read()
    if not grabbed:
        print "[INFO] Error de camara"
        break

    img2 = cv2.cvtColor(img2_rgb,cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img2,None)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    
    for m,n in matches:                                  ####################REVISAR QUE SE PUEDE MEJORAR
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

        cv2.polylines(img2,[np.int32(dst)],True,(255,0,0),3)

    else:
        print "No se encontraron suficientes puntos - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)

    img3 = draw_good_matches(img1,kp1,img2,kp2,good)
    cv2.imshow('SIFT',img3)
    key = cv2.waitKey(1)&0xFF
    if key == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()


