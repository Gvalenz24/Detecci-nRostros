import cv2
import numpy as np

FClas =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('oficina.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = FClas.detectMultiScale(gray,
    scaleFactor=1.1,#indica el % de reduccion que tendra la imagen
    minNeighbors= 5, #para poder reducir la seleccion a un solo rostro si los detectados esta muy cercanos
    minSize=(30,30),# medidas minimas para considerarse rostro
    maxSize=(200,200))#medidas maximas para considerarse rostro

for(x,y,w,h)  in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255),2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()