import cv2
import numpy as np
import torch
from ultralytics import YOLO
# from multiprocessing import Process, Queue
from collections import deque
from recorder import record_segment_from_frames, SizeLimitedFileHandler, SessionRecorder
import time
import logging
import os
import threading
import platform
import subprocess
import math
from queue import Queue, Empty, Full

USERNAME = "root"
PASSWORD = "wPro@3!4Gen*"
SCALE_PERCENTAGE = 100
FPS = 30

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global state
ui_frames = {}

# Initialize Recorder (Plug & Play)
# Usage: recorder.handle_frame(ip, frame) - does nothing if not recording
session_recorder = SessionRecorder()

def camera_worker_cv2(ip, queue, fps):
    # url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554"
    url = f"http://{USERNAME}:{PASSWORD}@{ip}/axis-cgi/mjpg/video.cgi?resolution=1920x1080&fps={fps}"
    print(f"Connecting to {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error(f"{ip} - Kameraya bağlanılamadı.")
        return

    while True:
        if not cap.isOpened():
            logger.warning(f"{ip} - Bağlantı koptu yeniden bağlanıyor...")
            cap.release()
            cap = cv2.VideoCapture(url)
            time.sleep(2)
            continue

        ret, frame = cap.read()
        if not ret:
            logger.warning(f"{ip} - Frame alınamadı. Yeniden bağlanmayı deniyor...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(url)
            continue
            
        # Resize if needed based on SCALE_PERCENTAGE
        if SCALE_PERCENTAGE != 100:
            width = int(frame.shape[1] * SCALE_PERCENTAGE / 100)
            height = int(frame.shape[0] * SCALE_PERCENTAGE / 100)
            frame = cv2.resize(frame, (width, height))
        
        timestamp = time.time()
        
        # Plug & Play Recording Hook
        # If recording is inactive, this returns immediately with minimal overhead
        session_recorder.handle_frame(ip, frame, fps=fps)

        # Put frame in queue
        # If queue is full, remove oldest item to make space (keep it real-time)
        if queue.full():
            try:
                queue.get_nowait()
            except Empty:
                pass
        
        try:
            queue.put((timestamp, frame), block=False)
        except Full:
            pass

def ui_worker(queues):
    grid_w, grid_h = 1280, 720
    window_name = "Camera Grid"
    
    while True:
        # Update latest frames from queues
        for ip, q in queues.items():
            try:
                # Get latest available frame
                while not q.empty():
                    timestamp, frame = q.get_nowait()
                    ui_frames[ip] = (timestamp, frame)
            except Empty:
                pass

        # Prepare Grid
        num_cameras = len(queues)
        if num_cameras == 0:
            time.sleep(0.1)
            continue

        cols = math.ceil(math.sqrt(num_cameras))
        rows = math.ceil(num_cameras / cols) if cols > 0 else 1
        
        cell_w = grid_w // cols
        cell_h = grid_h // rows
        
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        sorted_ips = sorted(queues.keys())
        
        for i, ip in enumerate(sorted_ips):
            if ip in ui_frames:
                timestamp, frame = ui_frames[ip]
                if frame is None: 
                    continue
                
                try:
                    resized = cv2.resize(frame, (cell_w, cell_h))
                    
                    row = i // cols
                    col = i % cols
                    
                    x1 = col * cell_w
                    y1 = row * cell_h
                    x2 = x1 + cell_w
                    y2 = y1 + cell_h
                    
                    # Clip if needed
                    if y2 > grid_h: y2 = grid_h
                    if x2 > grid_w: x2 = grid_w
                    
                    # Ensure dimensions match
                    h_curr = y2 - y1
                    w_curr = x2 - x1
                    if h_curr != resized.shape[0] or w_curr != resized.shape[1]:
                        resized = cv2.resize(frame, (w_curr, h_curr))

                    canvas[y1:y2, x1:x2] = resized
                    
                    # Draw IP label and timestamp
                    time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
                    cv2.putText(canvas, f"{ip} | {time_str}", (x1 + 10, y1 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Recording Indicator per Camera (on grid)
                    if session_recorder.get_status():
                        cv2.circle(canvas, (x2 - 30, y1 + 30), 10, (0, 0, 255), -1)
                        
                except Exception as e:
                    logger.error(f"Error resizing/drawing {ip}: {e}")

        # Global Recording Status Overlay
        if session_recorder.get_status():
            cv2.putText(canvas, "REC", (grid_w -110,40), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)

        cv2.imshow(window_name, canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Toggle recording
            if session_recorder.get_status():
                session_recorder.stop_session()
                logger.info("Recording STOPPED via UI")
            else:
                session_recorder.start_session()
                logger.info("Recording STARTED via UI")
        
        time.sleep(0.01)
            
    cv2.destroyAllWindows()
    # Ensure cleanup on exit
    session_recorder.stop_session()

if __name__ == "__main__":
    print("Starting the program...")
    camera_list = [
        "10.60.170.215",
        "10.60.170.216"
    ]
    
    queues = {}
    
    # Start camera threads
    for ip in camera_list:
        q = Queue(maxsize=2) # Small buffer to keep latency low
        queues[ip] = q
        t = threading.Thread(target=camera_worker_cv2, args=(ip, q, FPS), daemon=True)
        t.start()
        print(f"Started thread for {ip}")

    # Start UI thread
    # Note: cv2.imshow usually needs to run in the main thread on some systems (like macOS)
    # but on Windows it can often run in a separate thread if main waits.
    # To be safe and flexible, I'll run it as a thread but join it in main.
    ui = threading.Thread(target=ui_worker, args=(queues,), daemon=True)
    ui.start()
    print("Started UI thread")
    
    try:
        ui.join()
    except KeyboardInterrupt:
        print("Exiting...")
