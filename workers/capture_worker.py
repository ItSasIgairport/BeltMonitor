import cv2
import time
import logging
from queue import Empty, Full
from config_manager import ConfigManager

logger = logging.getLogger(__name__)

def camera_capture_worker(ip, ui_queue, inference_queue, recording_queue, event_queue, fps, session_recorder, shared_state):
    """
    Fast worker: Captures frames, handles recording/alarm buffering, and pushes to UI/Inference.
    Does NOT run YOLO.
    """
    config = ConfigManager()
    
    # Config extraction
    username = config.network.get('username', "root")
    password = config.network.get('password', "root")
    resolution = config.network.get('camera_resolution', "1920x1080")
    reconnect_delay = config.network.get('reconnect_delay', 2)
    
    source_type = config.network.get('source_type', 'stream')
    video_file_path = config.network.get('video_file_path', 'video/test.avi')

    if source_type == 'file':
        url = video_file_path
        logger.info(f"Using video file: {url}")
    else:
        url = f"http://{username}:{password}@{ip}/axis-cgi/mjpg/video.cgi?resolution={resolution}&fps={fps}"
        logger.info(f"Connecting to {url}")

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        logger.error(f"{ip} - Failed to connect to camera.")
        return

    # FPS Tracking
    fps_start_time = time.time()
    fps_frame_count = 0

    while True:
        if not cap.isOpened():
            logger.warning(f"{ip} - Connection lost, reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(url)
            time.sleep(reconnect_delay)
            continue

        ret, frame = cap.read()
        timestamp = time.time()

        # Dynamic Config Loading
        scale_percentage = config.processing.get('scale_percentage', 100)
        
        if not ret:
            if source_type == 'file':
                # Loop video file
                logger.info(f"{ip} - Video file ended, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                logger.warning(f"{ip} - Failed to grab frame. Retrying...")
                cap.release()
                time.sleep(reconnect_delay)
                cap = cv2.VideoCapture(url)
                continue
            
        # Resize if needed based on SCALE_PERCENTAGE
        if scale_percentage != 100:
            width = int(frame.shape[1] * scale_percentage / 100)
            height = int(frame.shape[0] * scale_percentage / 100)
            frame = cv2.resize(frame, (width, height))
            

        # FPS Calculation
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            curr_fps = fps_frame_count / elapsed
            if shared_state is not None and 'capture_fps' in shared_state:
                shared_state['capture_fps'][ip] = curr_fps
            fps_frame_count = 0
            fps_start_time = time.time()

        # --- 1. Push to Recording Queue (Fast) ---
        if recording_queue is not None:
            if session_recorder.get_status():
                 if recording_queue.full():
                    try:
                        recording_queue.get_nowait() # Drop oldest
                    except Empty:
                        pass
                 try:
                    recording_queue.put((timestamp, frame), block=False)
                 except Full:
                    pass

        # --- 1.5 Push to Event Queue (Always Active) ---
        if event_queue is not None:
            if event_queue.full():
                try:
                    event_queue.get_nowait()
                except Empty:
                    pass
            try:
                event_queue.put((timestamp, frame), block=False)
            except Full:
                pass

        # --- 2. Push to UI/Inference Queues (Non-blocking) ---
        
        # To Inference (Drop oldest if full)
        if inference_queue is not None:
             if inference_queue.full():
                try:
                    inference_queue.get_nowait()
                except Empty:
                    pass
             try:
                inference_queue.put((timestamp, frame), block=False)
             except Full:
                pass

        # To UI (Drop oldest if full)
        if ui_queue is not None:
            if ui_queue.full():
                try:
                    ui_queue.get_nowait() 
                except Empty:
                    pass
            try:
                ui_queue.put((timestamp, frame), block=False)
            except Full:
                pass

