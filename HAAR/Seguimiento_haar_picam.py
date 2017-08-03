#Importar las librerias necesarias
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import argparse
import cv2
import time


#Parsear los argumentos si es necesario
ap = argparse.ArgumentParser()
ap.add_argument("-v","--video",help="Opcional en caso de usar video")
args = vars(ap.parse_args())

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
    
#Cargamos la plantilla e inicializamos la webcam:
cascada_haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for frame in camara.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    #leemos un frame y lo guardamos
    imagen = frame.array

    #convertimos la imagen a blanco y negro
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
 
    #buscamos las coordenadas de los rostros (si los hay) y
    #guardamos su posicion
    caras = cascada_haar.detectMultiScale(gray, 1.3, 5)

    #Dibujamos un rectangulo en las coordenadas de cada rostro
    for (x,y,w,h) in caras:
        cv2.rectangle(imagen,(x,y),(x+w,y+h),(125,255,0),2)
        cv2.circle(imagen, ((x+w)/2, (y+h)/2), 7, (255, 255, 255), -1)
 
    #Mostramos la imagen
    cv2.imshow("Frame",imagen)
    key = cv2.waitKey(1)&0xFF
    rawCapture.truncate(0)
    
    if key == ord("q"):
        break

        
camara.release()
cv2.destroyAllWindows()




