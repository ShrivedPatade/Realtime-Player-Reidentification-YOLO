import cv2
import torch
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from collections import deque
import torch.nn.functional as F

# -------------------- CONFIG --------------------
MAX_HISTORY = 10
SIM_THRESHOLD = 0.65
DIST_THRESHOLD = 80  # in pixels
EMBEDDING_DIM = 512  # Simulated embedding dimension (match CNN output size later)
model_path = './proj/weights/best.pt'
video_path = './proj/input/15sec_input_720p.mp4'

# -------------------- INIT --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(model_path).to(device)
cap = cv2.VideoCapture(video_path)

known_players = {}  # {player_id: {'embeddings': deque, 'centroid': (x, y)}}
next_id = 0


# -------------------- FUNCTIONS --------------------
def cosine_distance(a, b):
    return 1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def euclidean_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def build_cost_matrix(embeddings, centroids):
    costs = []
    for det_emb, det_cent in zip(embeddings, centroids):
        row = []
        for pid, data in known_players.items():
            avg_emb = torch.stack(list(data['embeddings'])).mean(0)
            cost = cosine_distance(det_emb, avg_emb)
            spatial_dist = euclidean_distance(det_cent, data['centroid'])
            if spatial_dist > DIST_THRESHOLD:
                cost += 0.2  # penalize far matches
            row.append(cost)
        costs.append(row)
    return np.array(costs)

def assign_ids(detections, embeddings, centroids):
    global next_id

    if not known_players:
        for i in range(len(detections)):
            known_players[next_id] = {'embeddings': deque([embeddings[i]], maxlen=MAX_HISTORY), 'centroid': centroids[i]}
            detections[i]['id'] = next_id
            next_id += 1
        return

    cost_matrix = build_cost_matrix(embeddings, centroids)
    if len(cost_matrix) == 0 or cost_matrix.shape[1] == 0:
        return

    matched_rows, matched_cols = linear_sum_assignment(cost_matrix)
    assigned = set()

    for r, c in zip(matched_rows, matched_cols):
        if cost_matrix[r][c] < (1 - SIM_THRESHOLD):
            pid = list(known_players.keys())[c]
            known_players[pid]['embeddings'].append(embeddings[r])
            known_players[pid]['centroid'] = centroids[r]
            detections[r]['id'] = pid
            assigned.add(r)

    for i in range(len(detections)):
        if i not in assigned:
            known_players[next_id] = {'embeddings': deque([embeddings[i]], maxlen=MAX_HISTORY), 'centroid': centroids[i]}
            detections[i]['id'] = next_id
            next_id += 1


# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=device)[0]
    detections = []
    embeddings = []
    centroids = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        if label != 'player':
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Create dummy embedding using average color (replace with CNN later)
        crop_tensor = torch.tensor(crop).float().mean(dim=(0, 1)) / 255.0
        embedding = crop_tensor.repeat(EMBEDDING_DIM // 3)[:EMBEDDING_DIM]
        embeddings.append(embedding)
        centroids.append(((x1 + x2) // 2, (y1 + y2) // 2))
        detections.append({'box': (x1, y1, x2, y2)})

    if detections:
        assign_ids(detections, embeddings, centroids)

        for det in detections:
            x1, y1, x2, y2 = det['box']
            pid = det['id']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {pid}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Hungarian-Centroid Tracking', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
