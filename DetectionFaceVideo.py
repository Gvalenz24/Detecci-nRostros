import cv2
import numpy as np
cap = cv2.VideoCapture(0)
FClas =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame   , cv2.COLOR_BGR2GRAY)

    faces = FClas.detectMultiScale(gray,1.3,5)

    for(x,y,w,h)  in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255),2)

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()