import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 

#hacer 6 cortes antes de procesar imagenn??



def diferencia_imagenes (imagen1, imagen2):
    peso_a = 1
    peso_b = 2
  #Acomodar funcion para que devuelva ademas de la imagen, el nombre dentro de un array 
    diferencia = cv.addWeighted(imagen1, peso_a, -imagen2, peso_b, .5)
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


def pasar_hsv(imagen, nombre_imagen):
    imagen_convertida = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)
    nombre_salida_hsv = f"{os.path.splitext(nombre_imagen)[0]}_hsv.jpg"
    cv.imwrite(nombre_salida_hsv, imagen_convertida)

    h, s, v = cv.split(imagen_convertida)


    return imagen_convertida, nombre_salida_hsv


def encontrar_puntos_maximos(imagen_hsv, lineas):
    h, s, v = cv.split(imagen_hsv)
   
    v_maximos = []
    for x in lineas:
        columna_v = v[:, x]
        indice_max_v = np.argmax(columna_v) #devuelve el  indice del valor maximo 
        max_v = v[indice_max_v, x] 
        v_maximos.append((x, indice_max_v, max_v))
        # print (v_maximos)
        #Ojo! Revisar para ver si se puede encontrar mejor manera
    return v_maximos


def encontrar_recta (array_ordenadas):
    x = []
    y = []
    for xmax, index,  vmax in array_ordenadas :
        x.append(xmax)
        y.append(vmax)
    print ("x:",x)
    print ("y:",y)
    z = np.polyfit(x, y, 1)
    return z


x, y, anch, alto = 720, 200, 800, 300
muestras_puntos = [100, 200, 300, 400, 500, 600, 700]

# recortar_imagen(img1, x, y, anch, alto)
carpeta = '/home/ale/Documents/repos/Images Process/imagenes'

# #entorno de pruebas, no lo borres
imagenes_leidas = []
imagenes_hsv = []
array_imagenes = armado_rutas(carpeta)
for imagen_a , nombre_a in array_imagenes:
    imagen_recortada, nombre_original = recortar_imagen(imagen_a, x , y , anch, alto, nombre_a)
    imagenes_leidas.append((imagen_recortada, nombre_original))
    for imagen_recortada, nombre_original in imagenes_leidas:
        imagen_recortada_hsv, nombre_recortada_hsv = pasar_hsv(imagen_recortada, nombre_original)
        imagenes_hsv.append((imagen_recortada_hsv, nombre_recortada_hsv))

nombre_base_recortada = None
imagen_base_recortada = None



v_maximos = encontrar_puntos_maximos(imagen_recortada_hsv, muestras_puntos) 
print(v_maximos)
encontrar_recta(v_maximos)
#Agarrar indice y no  valor 


# carpeta_base = '/home/ale/Documents/repos/Images Process/base'

# imagen_base_array = armado_rutas(carpeta_base)
# for imagen_base, nombre_base in imagen_base_array:
#     imagen_base_recortada, nombre_base_recortada = recortar_imagen(imagen_base,x, y, anch, alto, nombre_base ) 
# #funcion para poder trabajar con la base y dejarla como variable global.

z = encontrar_recta(v_maximos)

x = [point[0] for point in v_maximos]
y = [point[2] for point in v_maximos]

#idiotaaaaa!! 

x_pred = np.linspace(min(x), max(x), 100)
y_pred = z[0] * x_pred + z[1]

plt.imshow(imagen_recortada_hsv, cmap='hsv')
plt.plot(x_pred, y_pred, color='red')

plt.scatter([point[0] for point in v_maximos], [point[1] for point in v_maximos], c='blue', label='Puntos máximos')

plt.show()
