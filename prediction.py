import numpy as np
from keras.models import model_from_json
import operator
import cv2 
import os,sys
import pyautogui



#load the model
json_file = open('model-bw.json', 'r')
model_json = json_file.read()
json_file.close()
load_model = model_from_json(model_json)

load_model.load_weights('model-bw.h5')
print('starting...')
cap = cv2.VideoCapture(0)
category = {0:'ZERO',5:'FIVE'} 

while True:
    _,frame = cap.read()
    x1 = 150
    y1 = 10
    x2 = 550
    y2 = 410

    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
  
    roi = frame[y1:y2,x1:x2]
    roi = cv2.resize(roi,(64,64))   
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _,test_img = cv2.threshold(roi,120,255,cv2.THRESH_BINARY)
    cv2.imshow('test image',test_img)

    res = load_model.predict(test_img.reshape(1,64,64,1))
    prediction = {'ZERO':res[0][0],
                  'FIVE':res[0][1]}
    
    prediction = sorted(prediction.items(),key = operator.itemgetter(1),reverse=True)
    numTest = prediction[0][0]
    cv2.putText(frame,numTest,(10,120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.imshow('frame',frame)
    if(numTest == 'FIVE'):
            pyautogui.press('space')    
    

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('q') : # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()


