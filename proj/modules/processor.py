# modules/processor.py
import torch
from ultralytics import YOLO

class FrameProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def annotate(self, frame):
        results = self.model(frame, device=self.device)[0]
        return results.plot()
