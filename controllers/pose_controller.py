import cv2
import base64
import numpy as np
from fastapi import APIRouter, UploadFile, File
from models.pose_detector import PoseDetector
from views.response_builder import build_response

router = APIRouter()
pose_detector = PoseDetector()

@router.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run detection
    img, keypoints = pose_detector.findPose(img)

    # Example: calculate left arm angle
    angle = None
    if len(keypoints) > 0:
        angle = pose_detector.findAngle(keypoints[0], 5, 7, 9)

    # Encode image back to base64
    _, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer).decode("utf-8")

    return build_response(img_str, angle)
