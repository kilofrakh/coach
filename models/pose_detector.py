import cv2
import numpy as np
import onnxruntime as ort

class PoseDetectorONNX:
    def __init__(self, model_path="yolov8n-pose.onnx", conf_threshold=0.5):
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.conf_threshold = conf_threshold

    def preprocess(self, image):
        # Resize & normalize (YOLO expects 640x640)
        img_resized = cv2.resize(image, (640, 640))
        img_input = img_resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return img_input, img_resized.shape[:2]

    def detect(self, image):
        input_data, _ = self.preprocess(image)

        # Run ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # YOLOv8 ONNX returns detections & keypoints
        detections = outputs[0]

        # Filter by confidence
        results = []
        for det in detections:
            if det[4] >= self.conf_threshold:  # confidence
                x1, y1, x2, y2, conf = det[:5]
                keypoints = det[5:].reshape(-1, 3)  # (x,y,confidence) per keypoint
                results.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(conf),
                    "keypoints": keypoints
                })

        return results
