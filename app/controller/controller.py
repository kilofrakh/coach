import cv2
import time
from model.model import PoseDetector, WorkoutCounter
from view.draw import View

class Controller:
    def __init__(self):
        self.detector = PoseDetector(detectionCon=0.8)
        self.workout = WorkoutCounter()
        self.cap = cv2.VideoCapture(0)
        self.pTime = 0

    def run(self):
        while True:
            success, img = self.cap.read()
            img = cv2.resize(img, (1366, 780))
            img = self.detector.findPose(img, draw=False)
            lmList = self.detector.findPosition(img, draw=False)

            if lmList:
                if lmList[31][2] + 50 > lmList[29][2] and lmList[32][2] + 50 > lmList[30][2]:
                    angle = self.detector.findAngle(11, 13, 15, img, draw=True)
                    self.detector.findAngle(12, 14, 16, img, draw=True)
                    self.detector.findAngle(27, 29, 31, img, draw=True)
                    self.detector.findAngle(28, 30, 32, img, draw=True)
                    self.workout.update(angle)
                    cTime = time.time()
                    fps = 1 / (cTime - self.pTime)
                    self.pTime = cTime
                    img = View.showWorkout(img, self.workout, fps)
                else:
                    img = View.showInstruction(img)
            else:
                img = View.showInstruction(img)

            cv2.imshow("Workout Tracker", img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()
