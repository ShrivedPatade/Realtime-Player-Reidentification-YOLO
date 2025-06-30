# modules/tracker.py
import torch
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

class PlayerTracker:
    def __init__(self, max_history=10, sim_threshold=0.6, dist_threshold=80):
        self.known_players = {}
        self.next_id = 0
        self.max_history = max_history
        self.sim_threshold = sim_threshold
        self.dist_threshold = dist_threshold

    def cosine_distance(self, a, b):
        return 1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def euclidean_distance(self, c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    def build_cost_matrix(self, embeddings, centroids):
        costs = []
        for det_emb, det_cent in zip(embeddings, centroids):
            row = []
            for pid, data in self.known_players.items():
                avg_emb = torch.stack(list(data['embeddings'])).mean(0)
                cost = self.cosine_distance(det_emb, avg_emb)
                spatial_dist = self.euclidean_distance(det_cent, data['centroid'])
                if spatial_dist > self.dist_threshold:
                    cost += 0.2
                row.append(cost)
            costs.append(row)
        return np.array(costs)

    def assign_ids(self, detections, embeddings, centroids):
        if not self.known_players:
            for i in range(len(detections)):
                self.known_players[self.next_id] = {
                    'embeddings': deque([embeddings[i]], maxlen=self.max_history),
                    'centroid': centroids[i]
                }
                detections[i]['id'] = self.next_id
                self.next_id += 1
            return

        cost_matrix = self.build_cost_matrix(embeddings, centroids)
        if len(cost_matrix) == 0 or cost_matrix.shape[1] == 0:
            return

        matched_rows, matched_cols = linear_sum_assignment(cost_matrix)
        assigned = set()

        for r, c in zip(matched_rows, matched_cols):
            if cost_matrix[r][c] < (1 - self.sim_threshold):
                pid = list(self.known_players.keys())[c]
                self.known_players[pid]['embeddings'].append(embeddings[r])
                self.known_players[pid]['centroid'] = centroids[r]
                detections[r]['id'] = pid
                assigned.add(r)

        for i in range(len(detections)):
            if i not in assigned:
                self.known_players[self.next_id] = {
                    'embeddings': deque([embeddings[i]], maxlen=self.max_history),
                    'centroid': centroids[i]
                }
                detections[i]['id'] = self.next_id
                self.next_id += 1
