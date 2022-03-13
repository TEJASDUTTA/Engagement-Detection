import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import sys
import os
from random import randint
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
no_of_images = 20

def captureImages(path, state):
    countImages=0
    count=0
    cap = cv2.VideoCapture(path)
    dir = "focused" if state == 0 else "distracted"
    
    while(countImages < no_of_images):
        ret, image = cap.read()
        if(ret == False):
            break
        
        # cv2.imshow("Image", image)

        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(rgb_image)

        left_eye_landmarks = []
        right_eye_landmarks = []

        if result.multi_face_landmarks:
            for facial_landmarks in result.multi_face_landmarks:
                j = 0
                for i in [68,107,236,117,336,298,346,456]:
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)

                    if j < 4:
                        left_eye_landmarks.append([x,y])
                    else:
                        right_eye_landmarks.append([x,y])
                    j += 1
                    # cv2.circle(image, (x,y), 2, (100, 100, 0), -1)

            count += 1

            if count%30 == 0:
                roi_left = image[left_eye_landmarks[0][1]:left_eye_landmarks[2][1], left_eye_landmarks[0][0]:left_eye_landmarks[2][0]]
                # roi_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
                # roi_left = cv2.equalizeHist(roi_left)
                roi_left[:][:][0] = cv2.equalizeHist(roi_left[:][:][0])
                roi_left[:][:][1] = cv2.equalizeHist(roi_left[:][:][1])
                roi_left[:][:][2] = cv2.equalizeHist(roi_left[:][:][2])
                # print(roi_left.shape)
                hL, wL, _ = roi_left.shape
                # print("Images stored so far:", countImages)
                # print(hL, wL)
                if hL >= 40 and wL >= 40:
                    cv2.imwrite("data/" + dir + "/right_eye" + str(randint(1, 10000)) + ".jpg", roi_left)
                    countImages += 1
                
                roi_right = image[right_eye_landmarks[0][1]:right_eye_landmarks[2][1], right_eye_landmarks[0][0]:right_eye_landmarks[2][0]]
                roi_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY)
                roi_right = cv2.equalizeHist(roi_right)


                hR, wR = roi_right.shape

                if hR >= 40 and wR >= 40:
                    cv2.imwrite("data/" + dir + "/left_eye" + str(randint(1, 10000)) + ".jpg", roi_right)
                    countImages += 1

        # cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(1)
    # close
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    directory = "video1"
    categories = ["focused", "distracted"]

    print("Count of videos: ", 2*len(os.listdir("video/focused")))
    i=1

    for category in categories:
        path = os.path.join(directory, category)

        for vid in os.listdir(path):
            vid_path = os.path.join(path, vid)
            if(category == "focused"):
                captureImages(vid_path, 0)
            else:
                captureImages(vid_path, 1)

            print("{0} video completed".format(i))
            i+=1