import cv2 as cv
import numpy as np
import os


def recortar_imagen(imagen, x, y, ancho, alto, nombre_original):
    if imagen is not None:
        # Recortar la región de interés (ROI)
        imagen_recortada = imagen[y:y+alto, x:x+ancho]

        # Construir el nombre del archivo de salida
        nombre_salida = f"{nombre_original}_recortada.jpg"

        # Guardar la imagen recortada
        cv.imwrite(nombre_salida, imagen_recortada)

        # Devolver la imagen recortada y el nombre del archivo de salida
        return imagen_recortada, nombre_salida
    else:
        print("No se pudo cargar la imagen.")
        return None, None

# Ruta de la carpeta que deseas recorrer
carpeta = "/home/ale/Documents/repos/Images Process/imagenes"

# Obtener la lista de archivos en la carpeta
a_recorrer = os.listdir(carpeta)

for nombre_archivo in a_recorrer:
    # Construir la ruta completa del archivo
    ruta_completa = os.path.join(carpeta, nombre_archivo)

    # Cargar la imagen
    imag = cv.imread(ruta_completa)

    # Obtener el nombre del archivo sin la extensión
    nombre_original, _ = os.path.splitext(nombre_archivo)
    # Coordenadas y dimensiones del recorte
    x, y, ancho, alto =  700, 0, 900 , 900

    # Llamar a la función de recorte
    recortar_imagen(imag, x, y, ancho, alto, nombre_original)

# Mostrar la primera imagen recortada como ejemplo
imagen_recortada, nombre_salida = recortar_imagen(imag, x, y, ancho, alto, nombre_original)
print (nombre_original)
print(nombre_salida)
cv.imshow("Imagen Recortada", imagen_recortada)
cv.waitKey(0)
cv.destroyAllWindows()


