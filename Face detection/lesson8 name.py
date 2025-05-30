import cv2,sys,os
import numpy as np

path = 'haarcascade_frontalface_default.xml'
facecascade = cv2.CascadeClassifier(path)

(images,labels,names,id) = ([],[],{},0)