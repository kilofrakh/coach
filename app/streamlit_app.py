import cv2
import time
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from model.model import PoseDetector, WorkoutCounter
from view.draw import View

st.set_page_config(page_title="Pose Workout Tracker", layout="wide")
st.title("🏋️ Pose Workout Tracker")
st.text("pushup")

detector = PoseDetector(detectionCon=0.8)
workout = WorkoutCounter()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoProcessorBase

class WorkoutProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # ✅ Downscale to save memory/CPU
        img = cv2.resize(img, (640, 360))

        self.frame_count += 1

        # ✅ Only process every 3rd frame
        if self.frame_count % 3 == 0:
            results = self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, 
                    results.pose_landmarks, 
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")
