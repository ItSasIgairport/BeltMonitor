import cv2
import torch
from ultralytics import YOLO
from queue import Queue
import threading
import os
import time
import logging

# Internal Modules
from config_manager import ConfigManager
from polygon_manager import PolygonManager
from alarm_manager import AlarmManager
from recorder import SessionRecorder
from logger import setup_logger

# Workers
from workers.capture_worker import camera_capture_worker
from workers.inference_worker import inference_worker
from workers.event_worker import event_worker
from workers.ui_worker import ui_worker

# Load Configuration & Logging
config = ConfigManager()
logger = setup_logger() # Configures root logger

def main():
    logger.info("Starting the program...")
    config.start_monitoring() # Start hot reloading
    
    # Global state (shared between threads)
    # Replaces globals: ui_frames (local to UI), latest_detections, processing_fps
    shared_state = {
        'detections': {},    # ip -> result object
        'capture_fps': {},   # ip -> float (renamed from 'fps' for clarity)
        'inference_fps': {},  # ip -> float
        'inference_time': {} # ip -> float (ms)
    }
    
    alarm_manager = AlarmManager()
    session_recorder = SessionRecorder()
    polygon_manager = PolygonManager()
    
    source_type = config.network.get('source_type', 'stream')
    
    if source_type == 'file':
        # Single worker for video file
        camera_list = ["video_file"]
        logger.info("Running in VIDEO FILE mode")
    else:
        # Load camera list from config for stream mode
        camera_list = config.network.get('cameras', [])
        if not camera_list:
            logger.warning("No cameras defined in config! Using default fallback.")
            camera_list = ["10.60.170.215", "10.60.170.216"]
        logger.info(f"Running in STREAM mode with cameras: {camera_list}")
    
    ui_queues = {}
    inference_queues = {}
    recording_queues = {} # Dedicated queues for disk writes
    event_queues = {} # Always-on queue for alarm buffering
    
    # Load YOLO model once in main thread
    logger.info("Loading YOLO model...")
    try:
        model_folder = config.processing.get('model_folder_path', "models")
        model_name = config.processing.get('model_name', "yolo11s-pose.pt")
        model_path = os.path.join(model_folder, model_name)
        
        model = YOLO(model_path)
        logger.info(f"YOLO model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        exit(1)
    
    # Start camera threads
    fps_val = config.processing.get('fps', 30)
    queue_size = config.ui.get('queue_size', 2)
    
    for ip in camera_list:
        uq = Queue(maxsize=queue_size) # Small buffer for UI
        iq = Queue(maxsize=2) # Small buffer for Inference
        rq = Queue(maxsize=60) # Buffer for recording (approx 2s @ 30fps)
        eq = Queue(maxsize=60) # Buffer for event analysis (approx 2s)
        
        ui_queues[ip] = uq
        inference_queues[ip] = iq
        recording_queues[ip] = rq
        event_queues[ip] = eq
        
        # Thread 1: Capture (Fast)
        t_cap = threading.Thread(
            target=camera_capture_worker, 
            args=(ip, uq, iq, rq, eq, fps_val, session_recorder, shared_state), 
            daemon=True
        )
        t_cap.start()
        
        # Thread 2: Inference (Slow)
        t_inf = threading.Thread(
            target=inference_worker, 
            args=(ip, iq, model, polygon_manager, alarm_manager, shared_state), 
            daemon=True
        )
        t_inf.start()

        # Register for recording
        session_recorder.register_camera(ip, rq, fps_val)
        
        # Thread 3: Event Processing (Always Active)
        t_evt = threading.Thread(
            target=event_worker, 
            args=(ip, eq, fps_val, alarm_manager), 
            daemon=True
        )
        t_evt.start()
        
        logger.info(f"Started decoupled workers for {ip}")

    # Start Alarm Manager thread
    am_thread = threading.Thread(target=alarm_manager.worker, daemon=True)
    am_thread.start()
    logger.info("Started Alarm Manager thread")

    # Start Performance Monitor thread
    pm_thread = threading.Thread(target=performance_monitor, args=(shared_state, 10), daemon=True)
    pm_thread.start()
    logger.info("Started Performance Monitor thread")

    # Start UI thread
    if config.ui.get('enable_ui', True):
        # UI Worker needs access to queues to consume frames
        ui = threading.Thread(
            target=ui_worker, 
            args=(ui_queues, polygon_manager, alarm_manager, session_recorder, shared_state), 
            daemon=True
        )
        ui.start()
        logger.info("Started UI thread")
        
        try:
            ui.join()
        except KeyboardInterrupt:
            logger.info("Exiting via KeyboardInterrupt...")
    else:
        logger.info("UI disabled by config")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting via KeyboardInterrupt...")
            
    # Ensure cleanup on exit
    session_recorder.stop_session()

def performance_monitor(shared_state, interval=10):
    """Logs performance statistics periodically for headless monitoring."""
    while True:
        time.sleep(interval)
        try:
            log_parts = ["Performance Stats:"]
            
            # Capture FPS
            if 'capture_fps' in shared_state and shared_state['capture_fps']:
                log_parts.append("  Capture FPS:")
                for ip, fps in shared_state['capture_fps'].items():
                    log_parts.append(f"    - {ip}: {fps:.2f}")
            
            # Inference FPS
            if 'inference_fps' in shared_state and shared_state['inference_fps']:
                log_parts.append("  Inference FPS:")
                for ip, fps in shared_state['inference_fps'].items():
                    log_parts.append(f"    - {ip}: {fps:.2f}")

            # Inference Time
            if 'inference_time' in shared_state and shared_state['inference_time']:
                log_parts.append("  Inference Time (ms):")
                for ip, ms in shared_state['inference_time'].items():
                    log_parts.append(f"    - {ip}: {ms:.2f} ms")
            
            # Only log if we have data
            if len(log_parts) > 1:
                logger.info("\n".join(log_parts))
                
        except Exception as e:
            logger.error(f"Error in performance monitor: {e}")

if __name__ == "__main__":
    main()
