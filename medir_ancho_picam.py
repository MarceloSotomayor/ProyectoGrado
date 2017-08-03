#Importar las librerias necesarias
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import argparse
import cv2
import time

#Definir los constantes para la distancia focal
DISTANCIA_CONOCIDA = 30
ANCHO_CONOCIDO = 11
PER = 15

distFocal = (DISTANCIA_CONOCIDA * PER )/ANCHO_CONOCIDO

#Parsear los argumentos si es necesario
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",help="Opcional en caso de usar video")
args = vars(ap.parse_args())

#Definir niveles de colores para la deteccion
colorBajo = np.array([49,50,50])
colorAlto = np.array([80,255,255])

#Definir si se usa video o camara web, validando algunas situaciones
if not args.get("video",False):
    camara = PiCamera()
    camara.resolution = (640,480)
    camara.framerate = 32
    rawCapture = PiRGBArray(camara,size=(640,480))
    
    print "[INFO] Calentando la camara ..."
    time.sleep(1)
    
else:
    camara = cv2.VideoCapture(args["video"])

#Definir las funciones necesarias, en este caso es la de medir distancia

def distancia_camara(anchoCon,distFocal,anchoPer):
    return (anchoCon*distFocal)/anchoPer


for frame in camara.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    imagen = frame.array

    hsv = cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, colorBajo, colorAlto)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    
    if len(cnts)>0:
        for c in cnts:
            if cv2.contourArea(c)<100:
                continue
            x,y,w,h = cv2.boundingRect(c)
            #distancia = distancia_camara(ANCHO_CONOCIDO,distFocal,w)
            cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.putText(imagen,str(distancia),(imagen.shape[1]-200,imagen.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,255,0),3)
            cv2.putText(imagen,str(w),(imagen.shape[1]-200,imagen.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,2.0,(0,255,0),3)
        
    cv2.imshow("Frame",imagen)
    key = cv2.waitKey(1)&0xFF
    rawCapture.truncate(0)
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
