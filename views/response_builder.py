import base64
import cv2

def build_response(img, count, progress):
    _, buffer = cv2.imencode(".jpg", img)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")
    return {
        "count": int(count),
        "progress": int(progress),
        "frame": frame_b64
    }
