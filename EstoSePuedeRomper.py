import cv2
import numpy as np


# redBajo2=np.array([175, 100, 20], np.uint8)
# redAlto2=np.array([179, 255, 255], np.uint8)
def test_mascara (imagen, color1, color2 ):
    frameHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    maskRed = cv2.inRange(frameHSV, color1, color2)
    # maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)
    
    maskRedvis = cv2.bitwise_and(imagen, imagen, mask= maskRed)        
  
    cv2.imshow('maskRed', maskRed)
    cv2.imshow('maskRedvis', maskRedvis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



img = cv2.imread('imagenes/20220204120649__800_1999983.jpg')

redBajo1 = np.array([135, 75, 20], np.uint8)
redAlto1 = np.array([170, 255, 255], np.uint8)



test_mascara(img, redAlto1, redBajo1)
