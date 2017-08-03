#Importar las librerias necesarias
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
    camara = cv2.VideoCapture(1)
    print "[INFO] Calentando la camara ..."
    time.sleep(2)
    camara.set(3,640)
    camara.set(4,480)
    time.sleep(2)

    if camara.isOpened():
        print "[INFO] Camara encendida"
else:
    camara = cv2.VideoCapture(args["video"])
    
#Cargamos la plantilla e inicializamos la webcam:
cascada_haar = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    #leemos un frame y lo guardamos
    (grabbed, frame) = camara.read()
    if not grabbed:
        print "[INFO] Error de camara"
        break
    #convertimos la imagen a blanco y negro
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    #buscamos las coordenadas de los rostros (si los hay) y
    #guardamos su posicion
    caras = cascada_haar.detectMultiScale(gray, 1.3, 5)

    #Dibujamos un rectangulo en las coordenadas de cada rostro
    for (x,y,w,h) in caras:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(125,255,0),2)
        cv2.circle(frame, ((x+w)/2, (y+h)/2), 7, (255, 255, 255), -1)
 
    #Mostramos la imagen
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)&0xFF

    if key == ord("q"):
        break

        
camara.release()
cv2.destroyAllWindows()




