# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:20:02 2022

@author: Asus
"""

from keras.models import load_model
from time import sleep 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import time
import  numpy as np
#import image
import datetime
import pyttsx3
mymodel=load_model('silkidea.h5')

cap=cv2.VideoCapture(0)
#face_cascade=cv2.CascadeClassifier('E:/seri/classifier/cascade.xml')
x=y=150
w=h=150
while True:
    _,img=cap.read()
    #face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=2)
    #for(x,y,w,h) in face:
    face_img = img[y:y+h,x:x+w]
    cv2.imwrite('temp.jpg',face_img)
    test_image=image.load_img('temp.jpg',target_size=(64,64,3))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    pred=mymodel.predict(test_image)
    if pred[0][0]==1:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        cv2.putText(img,"Affected",((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    elif pred[0][1]==1:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(img,'Pale Yellow',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    elif pred[0][2]==1:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(img,'Normal',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    
    datet=str(datetime.datetime.now())
    cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
      
    cv2.imshow('img',img)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()