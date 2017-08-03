#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time

from surf2 import CaracteristicaSURF

match = CaracteristicaSURF()

capture = cv2.VideoCapture(0)
time.sleep(1)
capture.set(3,640)
capture.set(4,480)
time.sleep(1)



while True:
    _,imagen = capture.read()
    exito,imagen_nueva = match.matching(imagen)
    cv2.imshow('SURF',imagen_nueva)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

capture.release()
cv2.destroyAllWindows()
