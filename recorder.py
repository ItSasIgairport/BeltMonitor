import cv2
import logging
import os
import time
import threading

logger = logging.getLogger(__name__)

class SizeLimitedFileHandler(logging.FileHandler):
    """
    Log handler that rotates files based on size.
    """
    def __init__(self, filename, max_bytes, encoding=None):
        super().__init__(filename, encoding=encoding)
        self.max_bytes = max_bytes

    def emit(self, record):
        if self.stream.tell() >= self.max_bytes:
            self.stream.close()
            # Simple rotation: delete and start fresh (or implement true rotation)
            open(self.baseFilename, 'w').close() 
            self.stream = self._open()
        super().emit(record)

def record_segment_from_frames(queue, fps, width, height, frame_skip):
    """
    Records warning segments from the queue.
    """
    while True:
        item = queue.get()
        if item is None:
            break
            
        ip = item['ip']
        frames = item['frames']
        
        # ... (existing segment recording logic if needed)
        pass

class SessionRecorder:
    """
    Plug-and-play recorder that manages simultaneous recording sessions for multiple cameras.
    """
    def __init__(self, base_dir="recordings"):
        self.base_dir = base_dir
        self.is_recording = False
        self.lock = threading.Lock()
        self.current_session_dir = None
        self.writers = {}  # ip -> cv2.VideoWriter

    def start_session(self):
        """Starts a new recording session."""
        with self.lock:
            if self.is_recording:
                return
            
            session_timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.current_session_dir = os.path.join(self.base_dir, f"session_{session_timestamp}")
            os.makedirs(self.current_session_dir, exist_ok=True)
            self.is_recording = True
            logger.info(f"🎥 Session Recording STARTED: {self.current_session_dir}")

    def stop_session(self):
        """Stops the current recording session and closes all writers."""
        with self.lock:
            if not self.is_recording:
                return
            
            self.is_recording = False
            # Close all writers
            for ip, writer in self.writers.items():
                if writer.isOpened():
                    writer.release()
            self.writers.clear()
            self.current_session_dir = None
            logger.info("⏹️ Session Recording STOPPED")

    def get_status(self):
        """Returns True if currently recording."""
        # No lock needed for simple boolean read (atomic in Python)
        return self.is_recording

    def handle_frame(self, ip, frame, fps=5.0):
        """
        Writes the frame if recording is active. 
        Initializes writer for the IP if it doesn't exist.
        Safe to call continuously.
        """
        if not self.is_recording:
            return

        with self.lock:
            # Double-check inside lock to avoid race conditions
            if not self.is_recording:
                return
            
            if ip not in self.writers:
                # Initialize writer for this camera
                h, w = frame.shape[:2]
                filename = os.path.join(self.current_session_dir, f"cam_{ip.replace('.', '_')}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
                
                if writer.isOpened():
                    self.writers[ip] = writer
                    logger.info(f"  -> Writing to {filename}")
                else:
                    logger.error(f"Failed to create writer for {ip}")
                    return

            # Write frame
            if ip in self.writers:
                self.writers[ip].write(frame)

