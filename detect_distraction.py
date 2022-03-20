import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import time
import sys
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import imutils
import dlib

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
distract_model = load_model("models/model_25img_t1_34v_colab.hdf5", compile=False)
img_width, img_height = 40, 40

emotion_offsets = (20, 40)
emotions = {
    0: { "emotion": "Angry",
        "color": (193, 69, 42) },
    1: { "emotion": "Disgust",
        "color": (164, 175, 49) },
    2: { "emotion": "Fear",
        "color": (40, 52, 155) },
    3: { "emotion": "Happy",
        "color": (23, 164, 28) },
    4: { "emotion": "Sad",
        "color": (164, 93, 23) },
    5: { "emotion": "Suprise",
        "color": (218, 229, 97) },
    6: { "emotion": "Neutral",
        "color": (108, 72, 200) }
}

faceLandmarks = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotionModelPath = "models/emotionModel.hdf5"
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]
        
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
        

    def shapePoints(self, shape):
        coords = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords


    def rectPoints(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    def facial_emotion(self, frame):
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(grayFrame, 0)
        for rect in rects:
            shape = predictor(grayFrame, rect)
            points = self.shapePoints(shape)
            (x, y, w, h) = self.rectPoints(rect)
            grayFace = grayFrame[y:y + h, x:x + w]
            try:
                grayFace = cv2.resize(grayFace, (emotionTargetSize))
            except:
                continue

            grayFace = grayFace.astype('float32')
            grayFace = grayFace / 255.0
            grayFace = (grayFace - 0.5) * 2.0
            grayFace = np.expand_dims(grayFace, 0)
            grayFace = np.expand_dims(grayFace, -1)
            emotion_prediction = emotionClassifier.predict(grayFace)
            for i in range(7):
                cv2.putText(frame, emotions[i]['emotion'] + " "+str(emotion_prediction[0][i]),
                            (10, 10+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)

            emotion_probability = np.max(emotion_prediction)
            if (emotion_probability > 0.36):
                emotion_label_arg = np.argmax(emotion_prediction)
                color = emotions[emotion_label_arg]['color']
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                         color,
                         thickness=2)
                cv2.rectangle(frame, (x + 20, y + h + 20), (x + 110, y + h + 40),
                              color, -1)
                cv2.putText(frame, emotions[emotion_label_arg]['emotion'],
                            (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            else:
                color = (255, 255, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            return emotion_probability

        return 0

    def here_it_goes(self):
        report = []
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            if(ret==False):
                break

            
            emotion = self.facial_emotion(frame)
            focused_level = self.detect_distraction(frame)

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