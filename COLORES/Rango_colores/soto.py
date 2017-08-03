import cv2
import numpy as np
import time

from picamera.array import PiRGBArray
from picamera import PiCamera

def nothing(x):
    pass

#CONSTANTES
limite_h  = 179
limite_sv = 255

camara = PiCamera()
camara.resolution = (640,480)
camara.framerate = 32
rawCapture = PiRGBArray(camara,size=(640,480))
time.sleep(1)

cv2.namedWindow('IMAGEN')

# CREANDO LAS BARRAS PARA LOS UMBRALES
cv2.createTrackbar('H_MAX','IMAGEN',0,limite_h,nothing)
cv2.createTrackbar('H_MIN','IMAGEN',0,limite_h,nothing)

cv2.createTrackbar('S_MAX','IMAGEN',0,limite_sv,nothing)
cv2.createTrackbar('S_MIN','IMAGEN',0,limite_sv,nothing)

cv2.createTrackbar('V_MAX','IMAGEN',0,limite_sv,nothing)
cv2.createTrackbar('V_MIN','IMAGEN',0,limite_sv,nothing)

# CREANDO EL SWITCH PARA ON/OFF
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'IMAGEN',0,1,nothing)

# CREANDO LA IMAGEN EN HSV

#hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

# CONDICIONES INICIALES
lower_color = np.array([0,50,50])
upper_color = np.array([10,255,255])

# CREACION DE MASCARA

#mask = cv2.inRange(hsv,lower_color,upper_color)

# IMAGEN FINAL MEDIANTE OPERACION AND
#res = cv2.bitwise_and(frame,frame,mask=mask)

for frame in camara.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    imagen = frame.array
    # CONVERSION DE FRAMES
    hsv = cv2.cvtColor(imagen,cv2.COLOR_BGR2HSV)
    # MOSTRAMOS LA IMAGEN
    
    # SALIMOS CON TECLA ESC
    

    # OBTENER LOS VALORES DE LAS PALETAS
        
    lower_h = cv2.getTrackbarPos('H_MIN','IMAGEN')
    upper_h = cv2.getTrackbarPos('H_MAX','IMAGEN')

    lower_s = cv2.getTrackbarPos('S_MIN','IMAGEN')
    upper_s = cv2.getTrackbarPos('S_MAX','IMAGEN')

    lower_v = cv2.getTrackbarPos('V_MIN','IMAGEN')
    upper_v = cv2.getTrackbarPos('V_MAX','IMAGEN')
    
    s = cv2.getTrackbarPos(switch,'IMAGEN')

    if s == 0:
        mask = imagen
    else:
        lower_color = np.array([lower_h,lower_s,lower_v])
        upper_color = np.array([upper_h,upper_s,upper_v])
        mask = cv2.inRange(hsv,lower_color,upper_color)
        #res = cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('IMAGEN',mask)
    key = cv2.waitKey(1)&0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
