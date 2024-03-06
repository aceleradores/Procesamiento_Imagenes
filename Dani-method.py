import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 


#convertir unt8

def diferencia_imagenes (imagen1, imagen2):
    imagen_a = np.uint8(imagen1)
    imagen_b = np.uint8(imagen2)
    peso_a = 1
    peso_b = 3

    # diferencia = cv.subtract(imagen_a, imagen_b)
    diferencia = cv.addWeighted(imagen_a, peso_a, -imagen_b, peso_b, 1)
    cv.imwrite("diferencia.jpg" , diferencia)
    return diferencia


def convertir_gris (imagen):
    gray_image = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY) 
    cv.imwrite("gris2.jpg", gray_image)
    return gray_image
    

def imagen_binary (imagen1):
    imagen_gris = cv.cvtColor(imagen1, cv.COLOR_BGR2GRAY) 
    # blur = cv.GaussianBlur(imagen_gris,(5,5),0)
    ret3,th3 = cv.threshold(imagen_gris,90,255,cv.THRESH_BINARY)
    cv.imwrite("binario2.jpg", th3)
    return th3

def imagen_limpia (imagen_limpia):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.erode(imagen_limpia,kernel,iterations = 1)
    cv.imwrite("imagenlimpia2.jpg", erosion)
    return erosion

def convertir_hsv (imagen_hsv, nombre):
    image_convertida = cv.cvtColor(imagen_hsv, cv.COLOR_BGR2HSV)
    cv.imwrite(f"{os.path.splitext(nombre)}-hsv.jpg", image_convertida)
    cv.imshow("v", v )
    # cv.waitKey(0)
    return image_convertida

    #hay que convertir a hsv 
    # hacer una mascara para trabajar solamente en el ¿h? => revisar el canales para


# CUIDADO CON LA GESTION DE DATATYPE
#El problema es que estas mandando un ndarray cuando
#cuando deberias mandar un string 
def recortar_imagen(ruta_imagen, x, y, ancho, alto):
    if ruta_imagen is not None:
        imagen_leida = cv.imread(ruta_imagen)
        imagen_recortada = imagen_leida[y:y+alto, x:x+ancho]
        nombre_salida = f"{os.path.splitext(ruta_imagen)}_recortada.jpg"
        print (nombre_salida)
        cv.imwrite(nombre_salida, imagen_recortada)
        return imagen_recortada, nombre_salida

img1 = '/home/ale/Documents/repos/Images Process/base.jpg'


# x, y, anch, alto = 700, 0, 900, 900
# recortar_imagen(img1, x, y, anch, alto)
carpeta = '/home/ale/Documents/repos/Images Process/imagenes'
carpeta_origen = os.listdir(carpeta)


for images in carpeta_origen:
    x, y, anch, alto = 700, 0, 900, 900
    imagen_recortada, nombre_salida = recortar_imagen(images, x, y, anch, alto)
    
    imagen_hsv= convertir_hsv(imagen_recortada, nombre_salida)
    cv.imshow (imagen_hsv)

