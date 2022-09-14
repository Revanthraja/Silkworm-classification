# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:54:39 2022

@author: Toshiba

"""

import cv2
import numpy as np
import pyttsx3
from keras.models import load_model
from PIL import Image
image_directory='b1/'
model=load_model('Silkworm10Epoch.h5')
image=cv2.imread('E:/images/Genrate-img/Test/diseased/_0_1560141.jpg')
img=Image.fromarray(image)

img=img.resize((64,64))
img=np.array(img)
input_img=np.expand_dims(img,axis=0)

result=model.predict(input_img)
engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
def brain():
    if result==0:
        speak(f"This brain is not affected by tumor sir!")
    else:
        speak(f"This brain affected by  tumor sir!")
brain()
    
    
    
print(result)
if result==0:
    print("This is affected")
else:
    print("This is not affected")
#else:
    #print("car image")
