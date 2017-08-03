#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

__autor__ = "Juan Marcelo Sotomayor Segales"

class CaracteristicaSURF:

    def __init__(self, train_imagen = "train2.png"):

        #Inicializar SURF
        self.hessiano_minimo = 400
        self.SURF            = cv2.SURF(self.hessiano_minimo)

        #Imagen de entrenamiento
        self.imagen_obj = cv2.imread(train_imagen, cv2.CV_8UC1)

        #Validacion de entrada
        if self.imagen_obj is None:
            print "No se encontro la imagen de entrenamiento" + train_imagen
            raise SystemExit

        #Calcular tamano de imagen y puntos caracteristicos de SURF
        self.tamano_imagen = self.imagen_obj.shape[:2]
        self.key_train, self.desc_train = self.SURF.detectAndCompute(self.imagen_obj,None)

        #Inicializar FLANN (Libreria Rapida para Aproximar los Vecinos mas Proximos)
        FLANN_INDEX_KDTREE = 0
        p_indice   = dict(algorithm = FLANN_INDEX_KDTREE) 
        p_busqueda = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(p_indice,p_busqueda)

        #Inicializar seguimiento
        self.ultimo_hinv = np.zeros((3,3))
        self.no_exitoso_frames = 0
        self.no_exitoso_f_max  = 5
        self.error_max_h = 50.

    def matching(self, imagen):

        #Crear una copia de la imagen normal y guardar su tamano
        imagen_in = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        tamano_in = imagen_in.shape[:2]

        #Extraccion de caracteristicas de la imagen_in (entrante) usando SURF
        key_in, desc_in = self._extraer_caracteristicas(imagen_in)

        #Emparejamiento de caracteristicas
        buenos_puntos = self._match_caracteristicas(desc_in)

        #Rechazar puntos inservibles
        if len(buenos_puntos) < 4:
            self.no_exitoso_frames = self.no_exitoso_frames + 1
            return False, imagen

        #Deteccion de puntos calculando matriz Homeografica entre puntos de entrenamiento y entrada
        dst_bordes = self._detectar_puntos_bordes(key_in, buenos_puntos)

        
        #Rechazar puntos, encontrar el area del cuadrilatero
        #area = 0
        #for i in xrange(0,4):
        #    sig_i = (i+1)%4
        #    area  = area + (dst_bordes[i][0]*dst_bordes[sig_i][1]-dst_bordes[i][0]*dst_bordes[sig_i][0])/2.

        
        #Ajustar coordenas x (columnas) de los puntos de borde
        dst_bordes = [(np.int(dst_bordes[i][0] + self.tamano_imagen[1]), np.int(dst_bordes[i][1])) for i in xrange(len(dst_bordes))]

        #Esquema de puntos del entrenamiento en la de entrada
        imag_flann = self.dibujar_buenos_p(self.imagen_obj, self.key_train, imagen_in, key_in, buenos_puntos)
        #for i in xrange(0, len(dst_bordes)):
        #    cv2.line(imag_flann, dst_bordes[i], dst_bordes[(i+1)%4],(0,255,0),3)
        cv2.polylines(imag_flann,[np.int32(dst_bordes)],True,(255,0,0),3)
        #Objeto frontal
        #[Hinv,dst_tamano] = self._warp_puntoskey(buenos_puntos,key_in,tamano_in)

        
        #Resetear puntos
        #self.no_exitoso_frames = 0
        #self.ultimo_h = Hinv

        #img_out = cv2.warpPerspective(imagen_in, Hinv, dst_tamano)
        #img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2RGB)

        return True, imag_flann

    def _extraer_caracteristicas(self, imagen_f):
        return self.SURF.detectAndCompute(imagen_f, None)
    
    def _match_caracteristicas(self, desc_imagen):

        #Encontrar las 2 mejores descripciones
        matches = self.flann.knnMatch(self.desc_train, desc_imagen, k=2)
        buenos_puntos = filter(lambda x:x[0].distance < 0.7*x[1].distance, matches)
        buenos_puntos = [buenos_puntos[i][0] for i in xrange (len(buenos_puntos))]
        return buenos_puntos

    def _detectar_puntos_bordes(self, key_imagen, buenos_puntos):

        #Encontrar Homeografico usando RANSAC (Random Sample Conseus)
        src_puntos = [self.key_train[buenos_puntos[i].queryIdx].pt for i in xrange(len(buenos_puntos))]
        dst_puntos = [key_imagen[buenos_puntos[i].trainIdx].pt for i in xrange(len(buenos_puntos))]
        H, _ = cv2.findHomography(np.array(src_puntos), np.array(dst_puntos), cv2.RANSAC)

        src_bordes = np.array([(0,0),(self.tamano_imagen[1],0),(self.tamano_imagen[1],self.tamano_imagen[0]),(0,self.tamano_imagen[0])],dtype=np.float32)

        dst_bordes = cv2.perspectiveTransform(src_bordes[None,:,:],H)
        dst_bordes = map(tuple,dst_bordes[0])

        return dst_bordes

    def _warp_puntoskey(self, buenos_puntos, key_frame, tamano_frame):

        dst_tamano = (tamano_frame[1], tamano_frame[0])
        escala_f   = 1./self.tamano_imagen[0]*dst_tamano[1]/2.
        bias_f     = dst_tamano[0]/4.
        escala_c   = 1./self.tamano_imagen[1]*dst_tamano[0]*3/4.
        bias_c     = dst_tamano[1]/8.

        src_puntos = [key_frame[buenos_puntos[i].trainIdx].pt for i in xrange(len(buenos_puntos))]
        dst_puntos = [self.key_train[buenos_puntos[i].queryIdx].pt for i in xrange(len(buenos_puntos))]
        dst_puntos = [[x*escala_f+bias_f,y*escala_c+bias_c] for x,y in dst_puntos]

        Hinv, _ = cv2.findHomography(np.array(src_puntos),np.array(dst_puntos),cv2.RANSAC)

        return [Hinv, dst_tamano]

    def dibujar_buenos_p(self,img1,kp1,img2,kp2,matches):
        #Crear imagen para concatenar
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

        out = np.zeros((max([rows1, rows2]), cols1+cols2, 3), dtype='uint8')
        
        out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])

        out[:rows2, cols1:cols1+cols2, :] = np.dstack([img2,img2,img2])

        radius = 4
        BLUE = (255,0,0)
        thickness = 1

        for m in matches:
            c1,r1 = kp1[m.queryIdx].pt
            c2,r2 = kp2[m.trainIdx].pt
            cv2.circle(out, (int(c1), int(r1)), radius, BLUE, thickness)
            cv2.circle(out, (int(c2)+cols1, int(r2)), radius, BLUE, thickness)

            cv2.line(out, (int(c1),int(r1)),(int(c2)+cols1,int(r2)), BLUE, thickness)

        return out           
