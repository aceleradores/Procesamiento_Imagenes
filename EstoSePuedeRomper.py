import cv2
import numpy as np
import os

#NO LO MODIFIQUES PORQUE ES MAS RAPIDO
# Ruta del directorio que contiene las imágenes
directorio_imagenes = r"/home/ale/Documents/repos/Images Process/fotos"

# Llamar a la función y obtener el resultado
resultado = conversor_rgb(directorio_imagenes)

# Hacer algo con el resultado, por ejemplo, mostrar la imagen resultante
if resultado is not None:
    cv2.imshow("Resultado Final", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


