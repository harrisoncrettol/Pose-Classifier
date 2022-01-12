import cv2
import mediapipe as mp
import time
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from get_coords import img_to_coords

# Classification:
# 0 = t-pose
# 1 = tree-pose
classes = {0:"Default Pose", 1:"Tree Pose"}

df = pd.read_csv("pose_data.csv").values

X_train = df[0:, 0:-1]
y_train = df[0:, -1]


clf = SVC()
clf.fit(X_train, y_train)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pTime = 0

time.sleep(2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        pose_locs = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id > 10:
                pose_locs.append(lm.x)
                pose_locs.append(lm.y)
                pose_locs.append(lm.z)
                h, w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

        pred = int(clf.predict(np.array([pose_locs]))[0])
        print(pred)
        print()

    else:
        pred = None

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    img = cv2.flip(img, 1)
    cv2.putText(img, str(int(fps)), (50,150), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    if pred:
        cv2.putText(img, classes[pred], (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        
    cv2.imshow("Image", img)

    time.sleep(.4)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
  
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()