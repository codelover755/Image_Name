import cv2,sys,os
import numpy as np

datasets = 'faceimages'
subdata = 'rahul'

path = os.path.join(datasets,subdata)

if not os.path.isdir(path):
    os.mkdir(path)

(width,height) = (500,500)

facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

frame = 1

while frame < 30:
    (_,im) = webcam.read()
    grey = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(grey,1.3,4)
    print(faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,0),2)
        face = grey[y:y + h,x:x + w]
        face_resize = cv2.resize(face,(width,height))
        cv2.imwrite('%s/%s.png'%(path,frame),face_resize)
    frame = frame + 1