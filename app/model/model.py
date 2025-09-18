import cv2
import mediapipe as mp
import math
import numpy as np

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=mode,
            smooth_landmarks=smooth,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = []

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, p1, p2, p3, img=None, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]

        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle > 180:
            angle = 360 - angle
        elif angle < 0:
            angle = -angle

        if draw and img is not None:
            cv2.circle(img, (x1, y1), 10, (64, 127, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (64, 127, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (64, 127, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 127, 64), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 127, 64), 3)
        return angle


class WorkoutCounter:
    def __init__(self):
        self.count = 0
        self.dir = 0
        self.per = 0
        self.bar = 0

    def update(self, angle):
        self.per = -1.25 * angle + 212.5
        self.per = max(0, min(100, self.per))  # clamp 0–100
        self.bar = np.interp(self.per, (0, 100), (650, 100))

        if self.per >= 95:
            if self.dir == 0:
                self.count += 0.5
                self.dir = 1
        elif self.per <= 5:
            if self.dir == 1:
                self.count += 0.5
                self.dir = 0
