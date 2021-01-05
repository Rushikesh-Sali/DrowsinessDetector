from threading import Thread


import cv2
import dlib
# import numpy as np
from scipy.spatial import distance
class eyes(Thread):

    def calculate_EAR(self,eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear_aspect_ratio = (A+B)/(2.0*C)
        return ear_aspect_ratio
class lips:
    def lip_distance(self,lip):
        A = distance.euclidean(lip[2], lip[10])
        B = distance.euclidean(lip[4], lip[8])
        C = distance.euclidean(lip[0], lip[6])
        lar_aspect_ratio = (A + B) / (2.0 * C)
        return lar_aspect_ratio


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        lip=[]
        # distance = lip_distance (face)

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        for n in range(48,60):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            lip.append((x,y))
            next_point = n+1
            if n == 59:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        object1 = eyes()
        object2 = lips()
        T1 = Thread( target = object1.calculate_EAR , args=(leftEye,))

        T2= Thread (target=object1.calculate_EAR, args =(rightEye,))
        T1.start ()
        T2.start ()


        left_ear = T1.join ()
        right_ear = T2.join ()
        left_ear1 = int(left_ear)
        right_ear1=int(right_ear)
        lip = object2.lip_distance(lip)
        EAR = (left_ear1+right_ear1)/2
        EAR = round(EAR,2)
        LAR =round(lip,2)
        if EAR<0.26 or LAR >0.4:
            cv2.putText(frame,"DROWSY",(20,100),
                cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
            cv2.putText(frame,"Are you Sleepy?",(20,400),
                cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
            print("Drowsy")
        print("EAR:",EAR)
        print ("LAR:", LAR)
        #
        # if LAR>1:
        #     cv2.putText (frame, "DROWSY", (20, 100),
        #         cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
        #     cv2.putText (frame, "Are you Sleepy?", (20, 400),
        #         cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        #     print ("Drowsy")
        # print ("LAR:",LAR)

    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()