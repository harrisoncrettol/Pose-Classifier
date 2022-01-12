import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('body-1-8-2022.mp4')

# Classification:
# 0 = t-pose
# 1 = tree-pose
def path_to_coords(path):
    img = cv2.imread(path)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    pose_locs = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id > 10:
                pose_locs.append(lm.x)
                pose_locs.append(lm.y)
                pose_locs.append(lm.z)
    return pose_locs

def img_to_coords(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    pose_locs = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id > 10:
                pose_locs.append(lm.x)
                pose_locs.append(lm.y)
                pose_locs.append(lm.z)
    return pose_locs





    