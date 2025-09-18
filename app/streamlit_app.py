import cv2
import time
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from app.model.model import PoseDetector, WorkoutCounter
from app.view.draw import View

st.set_page_config(page_title="Pose Workout Tracker", layout="wide")
st.title("🏋️ Pose Workout Tracker")
st.text("Real-time pose detection with Mediapipe + OpenCV + Streamlit")

detector = PoseDetector(detectionCon=0.8)
workout = WorkoutCounter()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class WorkoutProcessor(VideoProcessorBase):
    def __init__(self):
        self.pTime = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (960, 540))
        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            if lmList[31][2] + 50 > lmList[29][2] and lmList[32][2] + 50 > lmList[30][2]:
                angle = detector.findAngle(11, 13, 15, img, draw=True)
                detector.findAngle(12, 14, 16, img, draw=True)
                detector.findAngle(27, 29, 31, img, draw=True)
                detector.findAngle(28, 30, 32, img, draw=True)

                workout.update(angle)

                cTime = time.time()
                fps = 1 / (cTime - self.pTime) if self.pTime > 0 else 0
                self.pTime = cTime

                img = View.showWorkout(img, workout, fps)
            else:
                img = View.showInstruction(img)
        else:
            img = View.showInstruction(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="pose-tracker",
    mode="recvonly",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=WorkoutProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
