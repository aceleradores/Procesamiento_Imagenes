import cv2 as cv
import numpy as np

def encontrar_puntos_maximos(imagen_hsv, x_values):
    h, s, v = cv.split(imagen_hsv)
    max_s_points = []
    max_v_points = []

    for x in x_values:
        # Obtener la columna correspondiente al valor de x
        s_column = s[:, x]
        v_column = v[:, x]

        # Coordenadas del píxel con saturación máxima en la columna
        max_s_y = np.argmax(s_column)
        max_s_value = s[max_s_y, x]
        max_s_points.append((x, max_s_y, max_s_value))

        # Coordenadas del píxel con valor máximo en la columna
        max_v_y = np.argmax(v_column)
        max_v_value = v[max_v_y, x]
        max_v_points.append((x, max_v_y, max_v_value))

    return max_s_points, max_v_points

def dibujar_puntos_en_imagen(imagen, puntos, color):
    for punto in puntos:
        x, y, _ = punto
        cv.circle(imagen, (x, y), 3, color, -1)

# Ejemplo de uso
imagen = cv.imread("/home/ale/Documents/repos/Images Process/fotos ignore/20220204121951__800_1999983_recortada.jpg")
imagen_hsv = cv.cvtColor(imagen, cv.COLOR_BGR2HSV)

# Especifica los valores de x que te interesan
x_values = [100, 200, 300, 400, 500, 600,700]

max_s_points, max_v_points = encontrar_puntos_maximos(imagen_hsv, x_values)

# Dibujar puntos en la imagen
imagen_con_puntos = imagen.copy()
dibujar_puntos_en_imagen(imagen_con_puntos, max_v_points, (255, 0, 0))  # Color azul para puntos V

# Guardar o mostrar la imagen con los puntos
cv.imwrite("imagen_con_puntos.jpg", imagen_con_puntos)
cv.imshow("Imagen con Puntos", imagen_con_puntos)
cv.waitKey(0)
cv.destroyAllWindows()
