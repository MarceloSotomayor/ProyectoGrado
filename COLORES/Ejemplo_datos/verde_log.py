import cv2
import time
import numpy as np
import datetime
import os

i=0
muestreo=0.1
os.remove('opencv_log.csv')

set_time = 1
ancho = 320
alto  = 240

cap = cv2.VideoCapture(0)

time.sleep(set_time)
cap.set(3,ancho)
cap.set(4,alto)
cap.set(10,25)
time.sleep(set_time)

kernel = np.ones((3,3),np.uint8)

f=open('opencv_log.csv','a')
f.write("FECHA"+","+"EJEX"+","+"EJEY"+"\n")

while True:
    ret,frame = cap.read()

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_green = np.array([49,50,50])
    upper_green = np.array([80,255,255])

    mask = cv2.inRange(hsv,lower_green,upper_green)

    trans = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

    trans_fin = cv2.morphologyEx(trans,cv2.MORPH_CLOSE,kernel)
        
    moments = cv2.moments(trans_fin)
    area = moments['m00']
        
    if area>0:
        x = int(moments['m10']/moments['m00'])
        y = int(moments['m01']/moments['m00'])

    print(x)
    print(y)

    cv2.rectangle(frame,(x,y),(x+2,y+2),(0,0,255),2)

    now=datetime.datetime.now()
    timestamp = now.strftime("%Y/%m/%d %H:%M") 
    f.write(str(timestamp)+","+str(x)+","+str(y)+"\n")
    cv2.imshow('CAMARA',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
   
cap.release()
f.close()
cv2.destroyAllWindows()
print "FINALIZANDO" 
    
