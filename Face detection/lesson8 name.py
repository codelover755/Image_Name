import cv2,sys,os
import numpy as np

datasets = 'faceimages'

(images,labels,names,id) = ([],[],{},0)

for (subdirs,dirs,files) in os.walk(datasets):
    for subdirs in dirs:
        names[id] = subdirs
        fullpath = os.path.join(datasets,subdirs)
        for filename in os.listdir(fullpath):
            path = fullpath + "/" + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
            print(filename)
        id = id+1

(width,height) = (130,100)
(images,labels) = [np.array(lis) for lis in [images,labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)

codepath = 'haarcascade_frontalface_default.xml'
facecascade = cv2.CascadeClassifier(codepath)

webcam = cv2.VideoCapture(0)

while True:
    (_,im) = webcam.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    grey = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(grey,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = grey[y:y + h,x:x + w]
        face_resize = cv2.resize(face,(width,height))
        prediction = model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(im,str(names[prediction[0]]) + str(prediction[1]),(x,y),font,3,(0,0,255),2)                                      
    cv2.imshow("Face Recogniser",im)
    key = cv2.waitKey(0)
    if key == 27:
        break
