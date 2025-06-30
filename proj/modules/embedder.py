# modules/embedder.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchreid.utils import FeatureExtractor

# Embedder class for extracting player embeddings from video frames
# This class uses a pre-trained OSNet model to extract features from player crops
class Embedder:
    def __init__(self, osnet_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = FeatureExtractor(
            model_name='osnet_x0_25',
            model_path=osnet_path,
            device=str(device)
        )

    # Extract embeddings from the given frame and detections
    # Returns a list of embeddings, centroids, and valid detections
    def get_embeddings(self, frame, detections):
        embeddings = []
        centroids = []
        valid_detections = []

        for det in detections:
            x1, y1, x2, y2 = det['box']
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                emb = self.extractor(crop_rgb).squeeze(0).cpu()
            embedding = F.normalize(emb, dim=0)

            embeddings.append(embedding)
            centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
            valid_detections.append(det)

        return embeddings, centroids, valid_detections
