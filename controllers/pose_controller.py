import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from models.pose_detector import PoseDetector
from views.response_builder import build_response

router = APIRouter()

detector = PoseDetector(detectionCon=0.8)
count, dir, per = 0, 0, 0

@router.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    global count, dir, per

    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList):
        angle = detector.findAngle(img, 11, 13, 15)
        per = -1.25 * angle + 212.5
        per = max(0, min(100, per))

        if per >= 95 and dir == 0:
            count += 0.5
            dir = 1
        elif per <= 5 and dir == 1:
            count += 0.5
            dir = 0

    return build_response(img, count, per)
