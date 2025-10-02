import cv2
import numpy as np
from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_name="yolov8n-pose.pt"):
        """
        Initialize YOLOv8 Pose model.
        - model_name: you can swap with yolov8s/m/l/x-pose.pt depending on accuracy/speed tradeoff.
        """
        self.model = YOLO(model_name)

    def findPose(self, img, draw=True):
        """
        Run pose detection on an image.
        Returns:
            img (with keypoints drawn if draw=True)
            keypoints (list of [x,y] coordinates for each detected person)
        """
        results = self.model(img, verbose=False)[0]
        keypoints = []

        if results.keypoints is not None:
            keypoints = results.keypoints.xy.cpu().numpy()  # shape: (persons, 17, 2)

            if draw:
                for person in keypoints:
                    for x, y in person:
                        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

        return img, keypoints

    def findAngle(self, keypoints, p1, p2, p3):
        """
        Calculate angle between 3 keypoints.
        Args:
            keypoints: np.array of shape (17, 2) from YOLO
            p1, p2, p3: indices of points (e.g., 5, 7, 9 for left arm)
        Returns:
            angle in degrees
        """
        if keypoints is None or len(keypoints) < 17:
            return None

        x1, y1 = keypoints[p1]
        x2, y2 = keypoints[p2]
        x3, y3 = keypoints[p3]

        # Calculate angle
        a = np.array([x1 - x2, y1 - y2])
        b = np.array([x3 - x2, y3 - y2])
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        return angle
