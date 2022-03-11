import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import time
import sys

t1 = time.time()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# cap = cv2.VideoCapture("VideoFile.mp4")
cap = cv2.VideoCapture(0)

def captureImages(state, no_of_images, countImages=0, count=0):
    dir = "focused"
    if state == -1:
        dir = "distracted"
    
    while countImages < no_of_images:
        ret, image = cap.read()

        if ret != False:
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
                if count %2 == 0:
                    roi_left = image[left_eye_landmarks[0][1]:left_eye_landmarks[2][1], left_eye_landmarks[0][0]:left_eye_landmarks[2][0]]
                    roi_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
                    roi_left = cv2.equalizeHist(roi_left)
                    
                    hL, wL = roi_left.shape
                    print("Images stored so far:", countImages)
                    print(hL, wL)
                    if hL >= 40 and wL >= 40:
                        cv2.imwrite("data/" + dir + "/right_eye" + str(countImages) + ".jpg", roi_left)
                        countImages += 1
                    
                    roi_right = image[right_eye_landmarks[0][1]:right_eye_landmarks[2][1], right_eye_landmarks[0][0]:right_eye_landmarks[2][0]]
                    roi_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY)
                    roi_right = cv2.equalizeHist(roi_right)


                    hR, wR = roi_right.shape

                    if hR >= 40 and wR >= 40:
                        cv2.imwrite("data/" + dir + "/left_eye" + str(countImages) + ".jpg", roi_right)
                        countImages += 1
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(1)
    # close
    cap.release()
    cv2.destroyAllWindows()

state = 0
no_of_images = 50

if len(sys.argv) != 3:
    print("Usage: python get_eye_data.py no_of_images 0 (for focused) \n python get_eye_data no_of_images -1 (for distracted)")
    sys.exit()
else:
    no_of_images = int(sys.argv[1])
    state = int(sys.argv[2])

captureImages(state, no_of_images)

print("[*] Total time elapsed for", no_of_images, "images: ", (time.time() - t1))