# modules/reader.py
import threading
from queue import Queue, Empty

# FrameReader class for reading frames from a video capture object
# This class uses a separate thread to read frames and store them in a queue
class FrameReader:
    def __init__(self, cap, max_queue=5):
        self.cap = cap
        self.queue = Queue(maxsize=max_queue)
        self.stop_signal = object()
        self.thread = threading.Thread(target=self._reader)
        self.running = False

    # Private method to read frames from the video capture object
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

    # Start the frame reading thread
    def start(self):
        self.running = True
        self.thread.start()

    # Stop the frame reading thread and release the video capture object
    # Also clears the queue of any remaining frames
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

    # Get the next frame from the queue
    # If the queue is empty, returns a stop signal
    def get_frame(self):
        try:
            return self.queue.get(timeout=1)
        except Empty:
            return self.stop_signal

    # Check if the given frame is a stop signal
    # This is used to determine if the end of the video has been reached
    def is_stop(self, frame):
        return frame is self.stop_signal
