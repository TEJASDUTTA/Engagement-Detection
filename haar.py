import cv2
import numpy as np
from time import sleep
from keras.preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier(r'C:\Users\gupta\Downloads\ED\Engagement-Detection\cascades\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(r'C:\Users\gupta\Downloads\ED\Engagement-Detection\cascades\haarcascade_upperbody.xml')

def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = eye_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces)==0:
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((48,48), np.uint8), img
    return (x,w,y,h), roi_gray, img

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    # if np.sum([face]) != 0.0:
    #     roi = face.astype("float") / 255.0
    #     roi = img_to_array(roi)
    #     roi = np.expand_dims(roi, axis=0)

    #     # make a prediction on the ROI, then lookup the class
    #     preds = classifier.predict(roi)[0]
    #     label = class_labels[preds.argmax()]  
    #     label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)
    #     cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
    # else:
    #     cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)
        
    cv2.imshow('All', image)
    if cv2.waitKey(1) == 113: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()  