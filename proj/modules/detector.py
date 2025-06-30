# modules/detector.py
import torch
from ultralytics import YOLO

# Detector class for player detection using YOLOv8
# This class initializes the YOLO model and provides a method to detect players in a given frame
class Detector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

    # Detect players in the given frame
    # Returns a list of detections with bounding boxes for players
    def detect(self, frame):
        results = self.model(frame, device=self.device)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls]
            if label != 'player':
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({'box': (x1, y1, x2, y2)})
        return detections