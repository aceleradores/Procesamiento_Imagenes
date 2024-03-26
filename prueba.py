import cv2 as cv
import numpy as np
def extraer_valores_y_indice(imagen, lineas):
    valores_y = []  
    cambio_fase = []  
    
    for linea in lineas:  
        valor_anterior = None  
        for y, row in enumerate(imagen):
            if len(row) > linea:  
                pixel = row[linea]  
                valor_y = pixel[0]  
                valores_y.append(valor_y)  

                
                if valor_anterior is not None and valor_y != valor_anterior:
                    cambio_fase.append((valor_y, y))  # Guarda valor Y y su índice

                valor_anterior = valor_y  
            
    return valores_y, cambio_fase

imagen = cv.imread("/home/ale/Documents/repos/Images Process/20220204121951__800_1999983_recortada_hsv-gris-dilate-erode.jpg")


x_pos_especifico = [100]
valores_y_extraidos, cambios_de_fase = extraer_valores_y_indice(imagen, x_pos_especifico)



print("valores de y:", valores_y_extraidos)
print ("Cambio de fase " , cambios_de_fase)