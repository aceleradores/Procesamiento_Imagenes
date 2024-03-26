import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 
import skimage as skm

#hacer 6 cortes antes de procesar imagenn??



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

def convertir_gris (imagen, nombre):
    h, s, v = cv.split(imagen)
    ret3,th3 = cv.threshold(v ,0,255,cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    nombre_salida = f"{os.path.splitext(nombre)[0]}-gris.jpg"
    cv.imwrite(nombre_salida, th3)
    return th3, nombre_salida
    


def imagen_binary(imagen1, nombre):

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

# def imagen_limpia (imagen_limpia):
#     kernel = np.ones((5,5),np.uint8)
#     erosion = cv.erode(imagen_limpia,kernel,iterations = 1)
#     cv.imwrite("imagenlimpia2.jpg", erosion)
#     return erosion



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


def encontrar_limite(imagen, lineas):
    valores_y = []
    cambio_fase = []

    for linea in lineas:
        if not all(len(row) > linea for row in imagen):
            continue  # Salta la línea si alguna fila no tiene suficientes elementos

        valor_anterior = None
        for y, row in enumerate(imagen):
            pixel = row[linea]
            valor_y = pixel[0]
            valores_y.append(valor_y)

            if valor_anterior is not None and valor_y != valor_anterior:
                cambio_fase.append((valor_y, y))  # Guarda valor Y y su índice

            valor_anterior = valor_y

    return valores_y, cambio_fase


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




# x, y, anch, alto = 0, 150 , 640, 170
x, y, anch, alto = 720, 200, 800, 300

muestras_puntos = [100, 200, 300, 400, 500, 600]

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



for imagen_a , nombre_a in array_imagenes:
    imagen_recortada, nombre_original = recortar_imagen(imagen_a, x , y , anch, alto, nombre_a)
    imagenes_leidas.append((imagen_recortada, nombre_original))
    

nombre_base_final = None
imagen_base_final = None


imagen_base_array = armado_rutas(carpeta_base)
for imagen_base, nombre_base in imagen_base_array:
    imagen_base_recortada, nombre_base_recortada = recortar_imagen(imagen_base,x, y, anch, alto, nombre_base ) 
    imagen_base_final, nombre_base_final = pasar_hsv(imagen_base_recortada, nombre_base_recortada)

for imagen_diferencia, nombre_difrencia  in imagenes_leidas:
    imagen_diferencia, nombre_difrencia = pasar_hsv( imagen_diferencia, nombre_difrencia)
    imagenes_enderezadas.append((imagen_diferencia, nombre_difrencia))

for imagena, nombrea in imagenes_enderezadas:
    imagen_binaria, nombre_binario = convertir_gris(imagena,nombrea)
    imagenes_a_binario.append((imagen_binaria, nombre_binario))

for imagenbinaria, nombrebinaria in imagenes_a_binario:
    imagen_limpia, nombre_limpia   = imagen_binary(imagenbinaria, nombrebinaria )
    imagenes_ajustadas.append(imagen_limpia, nombre_limpia)

for imagerebinaria, nombrerebinario in imagenes_ajustadas:
    imagen_rebinaria, nombre_rebinario = imagen_binary(imagerebinaria, nombrerebinario)



# for imagen_recortada, nombre_recortado in imagen_recortadas:
#     v_maximos = encontrar_puntos_maximos(imagen_recortada_hsv, muestras_puntos) 
#     print(v_maximos)
#     ajustar_angulo(v_maximos,)


# plt.imshow(imagen_rotada, cmap='hsv')
# plt.plot(x_pred, y_pred, color='red')

# plt.scatter([point[0] for point in v_maximos], [point[1] for point in v_maximos], c='blue', label='Puntos máximos')

# plt.show()