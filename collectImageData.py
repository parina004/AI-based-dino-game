import os 
import cv2
import numpy as np
# create folder structure to store image data
if not os.path.exists('data'):
    os.makedirs('data')
    os.makedirs('data/train')
    os.makedirs('data/test')
    os.makedirs('data/train/0')
    os.makedirs('data/train/5')
    os.makedirs('data/test/0')
    os.makedirs('data/test/5')

mode = 'train'
directory = 'data/'+mode+'/'
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
   
    # get the count of existing images in the folder
    # create a count dictionary
    count = {'zero':len(os.listdir(directory+'/0')),
             'five':len(os.listdir(directory+'/5'))}

    # count of exisiting images in the folder 
    cv2.putText(frame,'MODE :-'+mode,(10,50),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame,'Image count',(10,70),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame,'ZERO :-'+str(count['zero']),(10,90),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame,'FIVE :-'+str(count['five']),(10,110),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    #generate ROI( region of intrest)
    # coordinates of roi
    x1 = 150
    y1 = 10
    x2 = 550
    y2 = 410
    # draw the rectange 
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
    # extract the roi
    roi = frame[y1:y2,x1:x2]
    roi = cv2.resize(roi,(128,128))
    cv2.imshow('frame',frame)
    # convert roi image to grayscale
    roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    # use the binary threshold method of open cv
    _,roi = cv2.threshold(roi,120,255,cv2.THRESH_BINARY)
    cv2.imshow('ROI',roi)
    interupt = cv2.waitKey(10)
    # store images 
    if interupt & 0xFF == ord('q'):
        break
    if interupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg',roi)
    if interupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg',roi)


cap.release()
cv2.destroyAllWindows()

