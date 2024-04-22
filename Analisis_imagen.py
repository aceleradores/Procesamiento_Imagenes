import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 
from scipy.optimize import curve_fit
import skimage as skm


def diferencia_imagenes (imagen1,nombre_1, imagen2,nombre_2):
    peso_a = 1
    peso_b = 2
    nombre_salida_1 = f"{os.path.splitext(nombre_1)[0]}"
    nombre_salida_2 = f"{os.path.splitext(nombre_2)[0]}"
    nombre_salida = f"{nombre_salida_1}-{nombre_salida_2}.jpg"
    diferencia = cv.addWeighted(imagen1, peso_a, -imagen2, peso_b, .5)
    cv.imwrite(nombre_salida , diferencia)
    return diferencia, nombre_salida


def obtener_puntos_eje_x(imagen):
    datos_x = []
    alto, ancho = imagen.shape[:2]
    paso = ancho // 49  
    datos_x.append(np.arange(0, ancho, paso))

    return datos_x

def promedio_vectores (imagen, linea) :
    y = imagen[:,:,2]
    print (y)
    

def otsu (imagen, nombre):
    h, s, v = cv.split(imagen)
    ret3,th3 = cv.threshold(v ,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    nombre_salida = f"{os.path.splitext(nombre)[0]}-otsu.jpg"
    cv.imwrite(nombre_salida, th3)
    return th3, nombre_salida

def convertir_gris (imagen, nombre):
    nombre_salida = f"{os.path.splitext(nombre)[0]}-gris.jpg"
    gris =  cv.cvtColor(imagen, cv.COLOR_RGB2GRAY)
    return gris, nombre_salida



def imagen_erode_dilate(imagen1, nombre):

    kernel = np.ones((5, 5), np.uint8)
    dilate_erode = cv.dilate(imagen1, kernel, iterations=1)
    dilate_erode_erode = cv.erode(dilate_erode, kernel, iterations=1)

    nombre_dilate_erode = f"{os.path.splitext(nombre)[0]}-dilate-erode.jpg"
    cv.imwrite(nombre_dilate_erode, dilate_erode_erode)


    erode_dilate = cv.erode(imagen1, kernel, iterations=1)
    imagen_convertida = cv.dilate(erode_dilate, kernel, iterations=1)
    nombre_imagen = f"{os.path.splitext(nombre)[0]}-erode-dilate.jpg"
    cv.imwrite(nombre_imagen, imagen_convertida)

    return imagen_convertida, nombre_imagen



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
 
    
    return mask_hsv_sumada, nombre_salida_hsv_sumada, 

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

# def graficar_datos(imagen, linea):
#     v = imagen[:,:,2]
    
#     valores_y = [v[:, y] for y in linea]
#     indices = [np.arange(len(v)) for _ in linea]
    
#     valores_y = np.concatenate(valores_y)
#     indices = np.concatenate(indices)
    
#     return indices, valores_y

# Graficar la distribución de la variable y comparar con una distribución de probabilidad adecuada.

def funcion_ajuste(x, A, sigma, offset,centro):
    return A/sigma * np.exp(-1/2*((x-centro)/sigma)**2) + offset


def encontrar_recta (array_ordenadas):
    x = []
    y = []
    for xmax, index,  vmax in array_ordenadas :
        x.append(xmax)
        y.append(index)
    print ("x:",x)
    print ("y:",y)
    z = np.polyfit(x, y, 1)
    return z

def ajustar_angulo (valores_maximos, imagen, nombre):
    z = encontrar_recta(valores_maximos)

    x = [point[0] for point in valores_maximos]
    y = [point[2] for point in valores_maximos]

    pendiente =z[0]

    angulo = np.degrees(np.arctan(pendiente))
    print ("el angulo es :",angulo)
    x_pred = np.linspace(min(x), max(x), 100)
    y_pred = z[0] * x_pred + z[1]
    imagen_rotada =skm.transform.rotate(imagen,angle =  angulo ,resize=False, mode='constant')
    nombre_salida = f"{os.path.splitext(nombre)[0]}rotada.jpg"
    return imagen_rotada, nombre_salida, x_pred, y_pred


def graficar_datos_imagen(imagen, linea):
    h, s, v = cv.split(imagen)
    valores_y = []
    indices = []
    for y in linea:
        valor_y = v[:, y]
        indices.extend(list(range(len(valor_y))))
        valores_y.extend(valor_y)
    return indices, valores_y
    


def graficar_datos(imagen, linea):
    v = imagen[:,:,2]
    valores_y = []
    indices = []
    
    for y in linea:
        valor_y = v[:, y]
        indices.extend(list(range(len(valor_y))))
        valores_y.extend(valor_y)
    return indices, valores_y

def encontrar_limite(imagen, lineas):
    valores_y = []
    cambio_fase = []

    for linea in lineas:
        if not all(len(row) > linea for row in imagen):
            continue  # Salta la línea si alguna fila no tiene suficientes elementos

        valor_anterior = None
        for y, row in enumerate(imagen):
            pixel = row[linea]
            valor_y = pixel
            valores_y.append(valor_y)

            if valor_anterior is not None and valor_y != valor_anterior:
                cambio_fase.append((linea, valor_y, y))  # Guarda valor Y y su índice

            valor_anterior = valor_y

    return valores_y, cambio_fase



# x, y, anch, alto = 0, 150 , 640, 170
x, y, anch, alto = 720, 200, 800, 300

muestras_puntos = [500]

# recortar_imagen(img1, x, y, anch, alto)
carpeta_base = "/home/ale/Documents/repos/Images Process/base"
# carpeta = r"C:\Users\aleja\Documents\Repositorios\Procesamiento_Imagenes\fotos"
# carpeta_base = r"C:\Users\aleja\Documents\Repositorios\Procesamiento_Imagenes\base"
carpeta = "/home/ale/Documents/repos/Images Process/imagenes"
# #entorno de pruebas, no lo borres
imagenes_leidas = []
imagenes_diferencias  =  []
imagenes_hsv = []
array_imagenes = armado_rutas(carpeta)
imagen_recortadas=[]
imagenes_ajustadas = []
imagenes_enderezadas = []
imagenes_a_binario = []

test = []

nombre_base_final = None
imagen_base_final = None

#Procesamineto de la base
imagen_base_array = armado_rutas(carpeta_base)
for imagen_base, nombre_base in imagen_base_array:
    imagen_base_recortada, nombre_base_recortada = recortar_imagen(imagen_base,x, y, anch, alto, nombre_base ) 
    imagen_base_final, nombre_base_final = pasar_hsv(imagen_base_recortada, nombre_base_recortada)

for imagen, nombre_imagen in array_imagenes:
    imagen_recortada, nombre_recortada = recortar_imagen(imagen, x, y, anch, alto, nombre_imagen)
    imagen_hsv, nombre_hsv = pasar_hsv(imagen_recortada, nombre_recortada)
    imagen_otsu, nombre_otsu = otsu(imagen_hsv, nombre_hsv)
    imagen_limpia, nombre_limpia = imagen_erode_dilate(imagen_otsu, nombre_otsu)
    valor_y, cambio_de_fase = encontrar_limite(imagen_limpia, muestras_puntos)

    y_medido = []
    for x in muestras_puntos:
        columna_v = imagen_hsv[:, x, 2]  # Obtener los valores del canal azul para la columna x
        y_medido.extend(columna_v)

    print (y_medido)
    # y_medido = np.array(y_medido)
    x_datos = range(len(y_medido))
    popt, pcov = curve_fit(funcion_ajuste, x_datos, y_medido)

    # # Graficar datos y ajuste
    # x_ajuste = np.linspace(min(muestras_puntos), max(muestras_puntos), 100)
    y_ajuste = funcion_ajuste(x_datos, *popt)

    plt.figure()
    plt.plot( y_medido, 'o', label='Datos')
    plt.plot(x_datos, y_ajuste, '-', label='Ajuste')
    plt.legend()
    plt.title(f'Ajuste para {nombre_imagen}')
    plt.show()
