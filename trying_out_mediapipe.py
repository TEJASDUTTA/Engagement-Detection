import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# cap = cv2.VideoCapture("VideoFile.mp4")
cap = cv2.VideoCapture(0)

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

EYES_COMBINED = LEFT_EYE + RIGHT_EYE


while True:
    ret, image = cap.read()

    if ret != False:
        height, width, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(rgb_image)

        eye_landmarks = []
        for facial_landmarks in result.multi_face_landmarks:
            # for i in [130, 243, 27, 23, 359, 463, 253, 257]:
            # for i in RIGHT_EYE:
            for i in [35, 122, 105, 119, 52]:
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)

                eye_landmarks.append((x,y))
                cv2.circle(image, (x,y), 2, (100, 100, 0), -1)
                # cv2.putText(image, str(i), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,255,0), 1)
        
        cv2.line(image, eye_landmarks[0], eye_landmarks[2], (0, 255, 0), 1)
        cv2.line(image, eye_landmarks[0], eye_landmarks[3], (0, 255, 0), 1)
        cv2.line(image, eye_landmarks[1], eye_landmarks[2], (0, 255, 0), 1)
        cv2.line(image, eye_landmarks[1], eye_landmarks[3], (0, 255, 0), 1)

    cv2.imshow("Image", image)
    cv2.waitKey(1)