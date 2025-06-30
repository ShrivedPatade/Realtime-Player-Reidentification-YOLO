# modules/reader.py
import threading
from queue import Queue, Empty

class FrameReader:
    def __init__(self, cap, max_queue=5):
        self.cap = cap
        self.queue = Queue(maxsize=max_queue)
        self.stop_signal = object()
        self.thread = threading.Thread(target=self._reader)
        self.running = False

    def _reader(self):
        while self.running:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                self.queue.put(self.stop_signal)
                break
            try:
                self.queue.put(frame, timeout=1)
            except:
                pass  # if queue is full, drop the frame

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.thread.join()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break

    def get_frame(self):
        try:
            return self.queue.get(timeout=1)
        except Empty:
            return self.stop_signal

    def is_stop(self, frame):
        return frame is self.stop_signal
