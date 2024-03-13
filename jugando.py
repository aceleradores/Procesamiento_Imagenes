import os
import cv2 as cv

def recortar_imagen(imagen_leida, x, y, ancho, alto, nombre_original):
    imagen_recortada = imagen_leida[y:y+alto, x:x+ancho]
    nombre_salida = f"{os.path.splitext(nombre_original)[0]}_recortada.jpg"
    print(f"Salio todo bien con {nombre_salida}")
    cv.imwrite(nombre_salida, imagen_recortada)
    return imagen_recortada, nombre_salida

def armado_rutas(directorio):
    imagenes = []

    directorio_recorrido = os.listdir(directorio)

    for archivo in directorio_recorrido:
        if archivo is not None:
            imagen = os.path.join(directorio, archivo)
            nombre_salida, _ = os.path.splitext(archivo)
            imagen_leida = cv.imread(imagen)

            if imagen_leida is not None:
                imagenes.append((imagen_leida, nombre_salida))
                print(f"Salio todo piola con {nombre_salida}")
            else:
                print(f"No se pudo cargar la imagen: {imagen}")
        else:
            print('Ale la cago')
    return imagenes

# Rutas de las imágenes
carpeta = '/home/ale/Documents/repos/Images Process/imagenes'

# Obtener lista de imágenes y nombres
imagenes = armado_rutas(carpeta)

# Recortar cada imagen en la lista
x, y, anch, alto = 700, 0, 900, 900
for imagen_a, nombre_a in imagenes:
    recortar_imagen(imagen_a, x, y, anch, alto, nombre_a)
