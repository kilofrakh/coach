import cv2
import time
import numpy as np
import streamlit as st
from model.model import PoseDetector, WorkoutCounter
from view.draw import View

st.set_page_config(page_title="Pose Workout Tracker", layout="wide")
st.title("🏋️ Pose Workout Tracker")
st.text("Real-time pose detection with Mediapipe + OpenCV + Streamlit")

frame_window = st.image([])
count_display = st.empty()
percent_display = st.empty()
fps_display = st.empty()

detector = PoseDetector(detectionCon=0.8)
workout = WorkoutCounter()
pTime = 0

# Camera input (browser webcam)
camera_frame = st.camera_input("🎥 Enable your webcam to start tracking")

if camera_frame:
    # Convert captured image (PIL) to OpenCV format
    file_bytes = np.asarray(bytearray(camera_frame.getbuffer()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

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
            fps = 1 / (cTime - pTime) if pTime > 0 else 0
            pTime = cTime

            img = View.showWorkout(img, workout, fps)

            count_display.metric("Reps", int(workout.count))
            percent_display.metric("Progress %", int(workout.per))
            fps_display.metric("FPS", int(fps))
        else:
            img = View.showInstruction(img)
    else:
        img = View.showInstruction(img)

    frame_window.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
else:
    st.info("👆 Allow camera access above to start workout tracking.")
