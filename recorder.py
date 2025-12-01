import cv2
import logging
import os
import time
import threading
from queue import Empty

logger = logging.getLogger(__name__)

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
    Starts/stops worker threads dynamically based on recording state.
    """
    def __init__(self, base_dir="recordings"):
        self.base_dir = base_dir
        self.is_recording = False
        self.lock = threading.Lock()
        self.current_session_dir = None
        self.writers = {}  # ip -> cv2.VideoWriter
        self.writer_stats = {} # ip -> {'start_time': float, 'frames_written': int}
        self.cameras = {} # ip -> {'queue': Queue, 'fps': float}
        self.active_threads = []

    def register_camera(self, ip, queue, fps):
        """Registers a camera configuration for recording."""
        with self.lock:
            self.cameras[ip] = {'queue': queue, 'fps': fps}
            logger.info(f"Registered camera for recording: {ip} (FPS: {fps})")

    def start_session(self):
        """Starts a new recording session and spawns worker threads."""
        with self.lock:
            if self.is_recording:
                return
            
            session_timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.current_session_dir = os.path.join(self.base_dir, f"session_{session_timestamp}")
            os.makedirs(self.current_session_dir, exist_ok=True)
            self.is_recording = True
            logger.info(f"🎥 Session Recording STARTED: {self.current_session_dir}")

            # Spawn threads for each registered camera
            self.active_threads = []
            for ip, config in self.cameras.items():
                t = threading.Thread(target=self._worker_loop, args=(ip, config['queue'], config['fps']), daemon=True)
                t.start()
                self.active_threads.append(t)
            logger.info(f"Started {len(self.active_threads)} recording workers.")

    def stop_session(self):
        """Stops the current recording session, waits for threads, and closes writers."""
        # 1. Signal threads to stop
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            logger.info("Stopping session, waiting for workers...")

        # 2. Wait for threads to finish
        for t in self.active_threads:
            if t.is_alive():
                t.join(timeout=2.0)

        self.active_threads = []

        # 3. Close writers
        with self.lock:
            for ip, writer in self.writers.items():
                if writer.isOpened():
                    writer.release()
            self.writers.clear()
            self.writer_stats.clear()
            self.current_session_dir = None
            logger.info("⏹️ Session Recording STOPPED")

    def get_status(self):
        """Returns True if currently recording."""
        return self.is_recording

    def _worker_loop(self, ip, queue, fps):
        """
        Dedicated worker loop for continuous recording.
        Only runs while self.is_recording is True.
        """
        logger.info(f"Recording worker started for {ip}")
        while self.is_recording:
            try:
                # Use timeout to check is_recording periodically if queue is empty
                timestamp, frame_orig = queue.get(timeout=1.0)
                
                # Double check status after getting frame
                if not self.is_recording:
                    break

                frame = frame_orig.copy()
                time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
                cv2.putText(frame, f"{ip} | {time_str}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                self._write_frame(ip, frame, fps=fps, timestamp=timestamp)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in recording worker for {ip}: {e}")
        
        logger.info(f"Recording worker finished for {ip}")

    def _write_frame(self, ip, frame, fps=5.0, timestamp=None):
        """
        Internal method to write frame to disk with synchronization logic.
        """
        if timestamp is None:
            timestamp = time.time()

        with self.lock:
            if not self.is_recording:
                return
            
            # Initialize writer if needed
            if ip not in self.writers:
                h, w = frame.shape[:2]
                filename = os.path.join(self.current_session_dir, f"cam_{ip.replace('.', '_')}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
                
                if writer.isOpened():
                    self.writers[ip] = writer
                    self.writer_stats[ip] = {'start_time': timestamp, 'frames_written': 0}
                    logger.info(f"  -> Writing to {filename}")
                else:
                    logger.error(f"Failed to create writer for {ip}")
                    return

            # Write frame with sync logic
            if ip in self.writers:
                stats = self.writer_stats[ip]
                
                # Calculate expected frames based on elapsed time
                elapsed = timestamp - stats['start_time']
                expected_frames = int(elapsed * fps)
                
                # Handle first frame case or potential clock skew
                if stats['frames_written'] == 0:
                    expected_frames = 1
                
                frames_to_write = expected_frames - stats['frames_written']
                
                # SYNC FIX: If frames_to_write < 1, it means we are ahead of schedule (too many frames).
                # We skip writing to avoid slow-motion (video playing slower than real-time).
                # Previously we forced it to 1, which caused every captured frame to be written 
                # even if capture FPS > video FPS.
                
                if frames_to_write > 0:
                    for _ in range(frames_to_write):
                        self.writers[ip].write(frame)
                        stats['frames_written'] += 1
