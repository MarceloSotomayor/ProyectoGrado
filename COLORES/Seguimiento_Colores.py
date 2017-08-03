#Importar las librerias necesarias
import numpy as np
import argparse
import cv2
import time

#Parsear los argumentos si es necesario
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",help="Opcional en caso de usar video")
args = vars(ap.parse_args())

#Definir niveles de colores para la deteccion
lower_green = np.array([49,50,50])
upper_green = np.array([80,255,255])

#Definir si se usa video o camara web, validando algunas situaciones
if not args.get("video",False):
    camara = cv2.VideoCapture(2)
    print "[INFO] Calentando la camara ..."
    time.sleep(1)
    camara.set(3,640)
    camara.set(4,480)
    time.sleep(1)

    if camara.isOpened():
        print "[INFO] Camara encendida"
else:
    camara = cv2.VideoCapture(args["video"])

#Kernel para las operaciones morfologicas de eliminacion de ruido
kernel = np.ones((3,3),np.uint8)

while True:
    (grabbed, frame) = camara.read()
    if not grabbed:
        print "[INFO] Error de camara"
        break
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv,lower_green,upper_green)
    
    trans = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    
    trans_fin = cv2.morphologyEx(trans,cv2.MORPH_CLOSE,kernel)

    moments = cv2.moments(trans_fin)

    area = moments['m00']
    
    if area>10:
        x = int(moments['m10']/moments['m00'])
        y = int(moments['m01']/moments['m00'])
        cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
        #print(x)
        #print(y)
        
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)&0xFF

    if key == ord("q"):
        break        

camara.release()
cv2.destroyAllWindows()
