# track.py (Main entry point)
import time
import cv2
from proj.modules.reader import FrameReader
from proj.modules.detector import Detector
from proj.modules.embedder import Embedder
from proj.modules.tracker import PlayerTracker

# -------------------- CONFIG --------------------
VIDEO_PATH = './proj/input/15sec_input_720p.mp4'
MODEL_PATH = './proj/weights/best.pt'
OSNET_PATH = './proj/osnet/osnet_x0_25_market1501.pth'
FPS_FALLBACK = 30

# -------------------- INIT --------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
frame_delay = int(1000 / fps)

reader = FrameReader(cap)
detector = Detector(MODEL_PATH)
embedder = Embedder(OSNET_PATH)
tracker = PlayerTracker()

reader.start()




while True:
    start = time.time()
    frame = reader.get_frame()
    if reader.is_stop(frame):
        break

    detections = detector.detect(frame)
    embeddings, centroids, filtered = embedder.get_embeddings(frame, detections)

    if filtered:
        tracker.assign_ids(filtered, embeddings, centroids)
        for det in filtered:
            x1, y1, x2, y2 = det['box']
            pid = det['id']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {pid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    print(f"Frame took {1000*(time.time() - start):.1f} ms")
    cv2.imshow('Hungarian-Centroid Tracking', frame)
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
reader.stop()
