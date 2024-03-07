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
    haz_alto = np.array([0 ,0 ,181], np.uint8)
    haz_bajo = np.array([179,255,255], np.uint8)
    image_convertida = cv.cvtColor(imagen_hsv, cv.COLOR_BGR2HSV)
    nombre_salida = f"{os.path.splitext(nombre)[0]}-hsv.jpg"
    mask_hsv= cv.inRange(image_convertida, haz_bajo, haz_alto)
    mask_hsv_sumada = cv.bitwise_and(imagen_hsv, imagen_hsv, mask= mask_hsv)
    print(f"Salio todo bien con {nombre_salida}")
    nombre_salida_hsv_mask = f"{os.path.splitext(nombre)[0]}-hsv-mask.jpg"
    nombre_salida_hsv_sumada = f"{os.path.splitext(nombre)[0]}-hsv-mask-sumada.jpg"
    cv.imwrite(nombre_salida_hsv_sumada, mask_hsv_sumada )
    cv.imwrite(nombre_salida_hsv_mask, mask_hsv)
    cv.imwrite(nombre_salida, image_convertida)
    # cv.imshow("v", v )
    # cv.waitKey(0)
    return image_convertida

    #hay que convertir a hsv 
    # hacer una mascara para trabajar solamente en el ¿h? => revisar el canales para


# CUIDADO CON LA GESTION DE DATATYPE
#El problema es que estas mandando un ndarray cuando
#cuando deberias mandar un string 
def recortar_imagen(imagen_leida, x, y, ancho, alto, nombre_original):
        imagen_recortada = imagen_leida[y:y+alto, x:x+ancho]
        nombre_salida = f"{os.path.splitext(nombre_original)[0]}_recortada.jpg"
        print (f"salio todo bien con {nombre_salida}")
        cv.imwrite(nombre_salida, imagen_recortada)
        return imagen_recortada, nombre_salida
#Hay que modificar que v recibe esta funcion.

def armado_rutas (directorio):
    imagenes = []

    directorio_recorrido = os.listdir(directorio)
    for archivo in directorio_recorrido:
        if archivo is not None:
            imagen = os.path.join(directorio, archivo)
            nombre_salida, _ = os.path.splitext(archivo) 
            imagen_leida= cv.imread(imagen)
            print(f"salio todo piola con {nombre_salida}")
            imagenes.append((imagen_leida, nombre_salida  ))
        else:
            print("Ale la cago") 
    return imagenes
#Devuelve un array con las imagenes


x, y, anch, alto = 720, 200, 800, 300
# recortar_imagen(img1, x, y, anch, alto)
carpeta = '/home/ale/Documents/repos/Images Process/imagenes'
array_imagenes = armado_rutas(carpeta)
for imagen_a , nombre_a in array_imagenes:
    imagen_recortada, nombre_original = recortar_imagen(imagen_a, x , y , anch, alto, nombre_a)
    diferencia_imagenes ()
carpeta_base = 'base'
imagen_base = armado_rutas(carpeta_base)
imagen_base, nombre_base = recortar_imagen() 
#funcion para poder trabajar con la base y dejarla como variable global. 