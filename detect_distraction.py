import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import time
import sys
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import imutils

# t1 = time.time()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
distract_model = load_model("model3.hdf5", compile=False)

# cap = cv2.VideoCapture("VideoFile.mp4")
cap = cv2.VideoCapture(0)

def detect_distraction():
    # dir = "focused"
    # if state == -1:
    #     dir = "distracted"
    
    while True:
        ret, image = cap.read()
        img_width, img_height = 40, 40
        if ret != False:
            height, width, _ = image.shape
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = face_mesh.process(rgb_image)

            left_eye_landmarks = []
            right_eye_landmarks = []

            probs = []
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
                        cv2.circle(image, (x,y), 2, (100, 100, 0), -1)

                # count += 1
                # if count %2 == 0:
                roi_left = image[left_eye_landmarks[0][1]:left_eye_landmarks[2][1], left_eye_landmarks[0][0]:left_eye_landmarks[2][0]]
                roi_left = cv2.resize(roi_left, (img_width, img_height))

                roi_left = roi_left.astype("float") / 255.0
                roi_left = img_to_array(roi_left)
                roi_left = np.expand_dims(roi_left, axis=0)

                predictionL = distract_model.predict(roi_left)
                probs.append(predictionL[0])
                # hL, wL, _L = roi_left.shape
                # print("Images stored so far:", countImages)
                # print(hL, wL)
                # if hL >= 60 and wL >= 60:
                #     cv2.imwrite("data/" + dir + "/right_eye" + str(countImages) + ".jpg", roi_left)
                    # countImages += 1
                    
                roi_right = image[right_eye_landmarks[0][1]:right_eye_landmarks[2][1], right_eye_landmarks[0][0]:right_eye_landmarks[2][0]]
                roi_right = cv2.resize(roi_right, (img_width, img_height))

                roi_right = roi_right.astype("float") / 255.0
                roi_right = img_to_array(roi_right)
                roi_right = np.expand_dims(roi_right, axis=0)

                predictionR = distract_model.predict(roi_right)
                probs.append(predictionR[0])

                print(probs)
                probs_mean = np.mean(probs)

                if probs_mean <= 0.5:
                    cv2.putText(image, "DISTRACTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(image, "FOCUSED",(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,0,255), 3, cv2.LINE_AA)
                # hR, wR, _R = roi_right.shape

                # if hR >= 60 and wR >= 60:
                #     cv2.imwrite("data/" + dir + "/left_eye" + str(countImages) + ".jpg", roi_right)
                    # countImages += 1
                
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(1)
    # close
    cap.release()
    cv2.destroyAllWindows()

# state = 0
# no_of_images = 50

# if len(sys.argv) != 3:
#     print("Usage: python get_data.py no_of_images 0 (for focused) \n python get_data no_of_images -1 (for distracted)")
#     sys.exit()
# else:
#     no_of_images = int(sys.argv[1])
#     state = int(sys.argv[2])

detect_distraction()

# print("[*] Total time elapsed for 100 images: ", (time.time() - t1))