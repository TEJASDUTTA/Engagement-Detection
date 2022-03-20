import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import time
import sys
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import imutils
# import dlib

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
distract_model = load_model("models/model_25img_t1_34v_colab.hdf5", compile=False)
img_width, img_height = 40, 40
        
class Engagement_Detection:
    def __init__(self):
        pass

    # Distraction Detector
    def predict(self, image, landmarks):
        roi = image[landmarks[0][1]:landmarks[2][1], landmarks[0][0]:landmarks[2][0]]
        if roi.any():
            roi = cv2.resize(roi, (img_width, img_height))
            
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = distract_model.predict(roi)
            return prediction

    def detect_distraction(self, image):
        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_image)

        left_eye_landmarks = []
        right_eye_landmarks = []

        probs = []
        probs_mean = 0
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
                    cv2.circle(image, (x,y), 5, (0, 255, 0), -1)


            draw_lines = [left_eye_landmarks, right_eye_landmarks]
            for i in draw_lines:
                for j in range(4):
                    cv2.line(image, i[j], i[(j+1)%4], (0, 255, 0), 1)
            
            try:
                probs.append(self.predict(image, left_eye_landmarks))
                probs.append(self.predict(image, right_eye_landmarks))

                # print(probs)
                probs_mean = np.mean(probs)

            except:
                pass

            if probs_mean <= 0.5:
                cv2.putText(image, "DISTRACTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0,0,255), 3, cv2.LINE_AA)

            else:
                cv2.putText(image, "FOCUSED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0,0,255), 3, cv2.LINE_AA)

        return probs_mean
        

    def here_it_goes(self):
        report = []
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            if(ret==False):
                break

            focused_level = self.detect_distraction(frame)
            # emotion = self.facial_emotion(frame)

            # index = self.get_concentration_index(focused_level, emotion)
            # report.append(index)

            cv2.imshow("Engagement Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        # report = np.array(report)
        # overall_index = np.mean(report)

        # if(overall_index > 0.65):
        #     print("You were highly engaged")
        # elif(overall_index > 0.25):
        #     print("You were nominally engaged")
        # else:
        #     print("Pay attention")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    obj = Engagement_Detection()
    obj.here_it_goes()