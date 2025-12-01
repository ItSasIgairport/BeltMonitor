import cv2
import numpy as np
import torch
from ultralytics import YOLO
# from multiprocessing import Process, Queue
from collections import deque
from recorder import record_segment_from_frames, SessionRecorder
from polygon_manager import PolygonManager
from config_manager import ConfigManager
from alarm_manager import AlarmManager
import time
import logging
import os
import threading
import platform
import subprocess
import math
import ctypes
from queue import Queue, Empty, Full

# Load Configuration
config = ConfigManager()
config.start_monitoring() # Start hot reloading

log_level_str = config.logging_config.get('level', "INFO")
LOG_LEVEL = getattr(logging, log_level_str.upper(), logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

# Global state
ui_frames = {}
latest_detections = {} # Stores RAW results objects: { ip: result }
processing_fps = {} # { ip: fps_value }
alarm_manager = AlarmManager()


# Initialize Recorder (Plug & Play)
# Usage: recorder.handle_frame(ip, frame) - does nothing if not recording
session_recorder = SessionRecorder()


def camera_capture_worker(ip, ui_queue, inference_queue, recording_queue, event_queue, fps):
    """
    Fast worker: Captures frames, handles recording/alarm buffering, and pushes to UI/Inference.
    Does NOT run YOLO.
    """
    # Config extraction
    username = config.network.get('username', "root")
    password = config.network.get('password', "root")
    resolution = config.network.get('camera_resolution', "1920x1080")
    reconnect_delay = config.network.get('reconnect_delay', 2)
    
    source_type = config.network.get('source_type', 'stream')
    video_file_path = config.network.get('video_file_path', 'video/test_video.mp4')

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
            processing_fps[ip] = curr_fps
            fps_frame_count = 0
            fps_start_time = time.time()

        # --- 1. Push to Recording Queue (Fast) ---
        # We push to recording queue immediately. 
        # If disk I/O is slow, this queue will fill up. We drop frames if full to avoid blocking capture.
        if recording_queue is not None:
            # Optimization: Only push to recording queue if recording is active?
            # NO - we push always so the worker can decide (and drain queue if needed).
            # Actually, if we don't push when off, the worker blocks on get() which is fine.
            # But if we start recording, we want fresh frames.
            # Let's keep pushing always for simplicity, or verify get_status() here.
            # If we check get_status() here, we save queue overhead when not recording.
            
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

def inference_worker(ip, inference_queue, model, polygon_manager):
    """
    Slow worker: Runs YOLO inference on latest available frame.
    Updates shared state 'latest_detections' with raw results.
    """
    while True:
        try:
            timestamp, frame = inference_queue.get()
            
            # Dynamic Config
            check_keypoints = config.processing.get('keypoints', [5, 6, 11, 12, 13, 14, 15, 16])
            conf_threshold = config.processing.get('model_confidence_threshold', 0.5)
            enable_yolo = config.processing.get('enable_yolo', True)

            # Retrieve polygon for the camera IP
            poly_data = polygon_manager.get_polygon(ip, 'mask')
            curr_frame = frame
            
            # Apply mask if exists
            if poly_data:
                points = poly_data['points']
                curr_frame = apply_mask(frame, points)
            
            # Run model if enabled
            results = None
            if enable_yolo:
                results = model.predict(curr_frame, verbose=False)
            
            # Retrieve belt polygon for logic check
            belt_poly_data = polygon_manager.get_polygon(ip, 'belt')

            current_intrusion = False

            # Store RAW results for UI to draw later
            if results:
                res = results[0]
                latest_detections[ip] = res # Store the object, not the plotted frame
                
                # --- Stage 1: Detection Logic ---
                if belt_poly_data:
                    belt_points = np.array(belt_poly_data['points'], dtype=np.int32)
                    
                    # Check keypoints for each detected person
                    if res.keypoints is not None and res.keypoints.xy is not None:
                        kpts_xy = res.keypoints.xy.cpu().numpy()
                        kpts_conf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
                        
                        for i, person_kpts in enumerate(kpts_xy):
                            person_on_belt = False
                            for kp_idx in check_keypoints:
                                x, y = person_kpts[kp_idx]
                                conf = kpts_conf[i][kp_idx] if kpts_conf is not None else 1.0
                                
                                if conf > conf_threshold and (x > 0 or y > 0):
                                    if cv2.pointPolygonTest(belt_points, (float(x), float(y)), False) >= 0:
                                        person_on_belt = True
                                        break
                            
                            if person_on_belt:
                                current_intrusion = True
                                break 

                # --- Push Event to Alarm Manager ---
                alarm_manager.add_event(ip, current_intrusion, timestamp)

        except Exception as e:
            logger.error(f"Error in inference logic for {ip}: {e}")

def event_worker(ip, event_queue, fps):
    """
    Dedicated worker for AlarmManager event processing (always active).
    Ensures event buffering and event clips are handled without blocking capture.
    """
    while True:
        try:
            timestamp, frame_orig = event_queue.get()
            
            # Always process frames for alarm manager (buffering/analysis)
            # We copy frame because AlarmManager might modify it (draw dots) or buffer it
            frame = frame_orig.copy()
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            cv2.putText(frame, f"{ip} | {time_str}", (10, 300), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 240, 0), 2)
            
            alarm_manager.process_frame(ip, frame, fps=fps, timestamp=timestamp)
            
        except Exception as e:
             logger.error(f"Error in event worker for {ip}: {e}")

def get_screen_resolution():
    try:
        if platform.system() == "Windows":
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception as e:
        logger.error(f"Failed to get screen resolution: {e}")
    return 1920, 1080 # Default fallback

def update_grid_layout(grid_layout, num_cameras, grid_w, grid_h):
    if num_cameras == 0:
        return

    cols = math.ceil(math.sqrt(num_cameras))
    rows = math.ceil(num_cameras / cols) if cols > 0 else 1
    
    cell_w = grid_w // cols
    cell_h = grid_h // rows
    
    grid_layout['cols'] = cols
    grid_layout['rows'] = rows
    grid_layout['cell_w'] = cell_w
    grid_layout['cell_h'] = cell_h

def draw_grid_cells(canvas, grid_layout, polygon_manager, grid_w, grid_h):
    sorted_ips = grid_layout['sorted_ips']
    cols = grid_layout['cols']
    cell_w = grid_layout['cell_w']
    cell_h = grid_layout['cell_h']
    
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

                # --- Alarm Visualization Border ---
                status = alarm_manager.get_status(ip)
                if status != 'SAFE':
                    border_color = (0, 255, 0) # Default
                    border_thickness = 2
                    
                    if status == 'WARNING':
                        border_color = (0, 255, 255) # Yellow
                        border_thickness = 4
                    elif status == 'ALARM':
                        border_color = (0, 0, 255) # Red
                        border_thickness = 8
                        
                    cv2.rectangle(resized, (0, 0), (w_curr-1, h_curr-1), border_color, border_thickness)

                # --- Draw Raw Frame First ---
                # Note: We are drawing detections ON TOP of this resized frame
                
                scale_x = w_curr / frame.shape[1]
                scale_y = h_curr / frame.shape[0]
                
                # --- Draw Pose Detections (Decoupled) ---
                # Retrieve latest result object
                if ip in latest_detections:
                    res = latest_detections[ip]
                    if res:
                        plotted = res.plot(img=frame) # Draw OLD boxes on NEW frame
                        resized = cv2.resize(plotted, (w_curr, h_curr))

                canvas[y1:y2, x1:x2] = resized

                # --- Draw Mask Polygon (Blue) ---
                mask_poly = polygon_manager.get_polygon(ip, 'mask')
                if mask_poly:
                    orig_points = mask_poly['points']
                    scaled_points = []
                    for pt in orig_points:
                        sx = int(pt[0] * scale_x) + x1
                        sy = int(pt[1] * scale_y) + y1
                        scaled_points.append([sx, sy])
                    
                    if len(scaled_points) > 0:
                        poly_cnt = np.array(scaled_points)
                        # Draw semi-transparent fill for mask
                        overlay = canvas.copy()
                        cv2.fillPoly(overlay, [poly_cnt], (255, 0, 0)) # Blue
                        cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas)
                        cv2.polylines(canvas, [poly_cnt], True, (255, 255, 0), 2) # Cyan outline

                # --- Draw Belt Polygon (Green/Red) ---
                belt_poly = polygon_manager.get_polygon(ip, 'belt')
                if belt_poly:
                    orig_points = belt_poly['points']
                    
                    scaled_points = []
                    for pt in orig_points:
                        sx = int(pt[0] * scale_x) + x1
                        sy = int(pt[1] * scale_y) + y1
                        scaled_points.append([sx, sy])
                        
                    if len(scaled_points) > 0:
                        poly_cnt = np.array(scaled_points)
                        
                        color = (0, 255, 0)
                        cv2.polylines(canvas, [poly_cnt], True, color, 2)
                
                # Draw IP label, timestamp, and FPS
                time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
                fps_text = ""
                if ip in processing_fps:
                    fps_text = f" | FPS: {processing_fps[ip]:.1f}"
                
                # Append status to label
                status_text = ""
                text_color = (0, 255, 255)
                
                status = alarm_manager.get_status(ip)
                if status != 'SAFE':
                     status_text = f" | {status}"
                     if status == 'ALARM': text_color = (0, 0, 255)
                     elif status == 'WARNING': text_color = (0, 255, 255)

                cv2.putText(canvas, f"{ip} | {time_str}{fps_text}{status_text}", (x1 + 10, y1 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Recording Indicator per Camera (on grid)
                if session_recorder.get_status():
                    cv2.circle(canvas, (x2 - 30, y1 + 30), 10, (0, 0, 255), -1)
                    
            except Exception as e:
                logger.error(f"Error resizing/drawing {ip}: {e}")

    # Global Recording Status Overlay
    if session_recorder.get_status():
        cv2.putText(canvas, "REC", (grid_w -110,40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2)
    
    # Edit Mode Indicator
    if config.ui.get('enable_polygon_ui', False):
        mode_str = polygon_manager.edit_mode.upper()
        cv2.putText(canvas, f"EDIT MODE: {mode_str}", (grid_w - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)

def handle_polygon_drawing_window(polygon_manager, grid_w, grid_h):
    drawing_state = polygon_manager.drawing_state
    if drawing_state['active']:
        ip = drawing_state['ip']
        win_name = drawing_state['window_name']
        
        if not drawing_state['window_initialized']:
                cv2.namedWindow(win_name)
                # Use callback from PolygonManager
                cv2.setMouseCallback(win_name, polygon_manager.mouse_callback_draw)
                drawing_state['window_initialized'] = True

        if ip in ui_frames:
            _, frame = ui_frames[ip]
            if frame is not None:
                # Use original resolution for saving, but scale for display
                drawing_state['frame_shape'] = frame.shape
                
                h_orig, w_orig = frame.shape[:2]
                scale = 1.0
                
                # Calculate scale if too big
                if w_orig > grid_w or h_orig > grid_h:
                    scale = min(grid_w / w_orig, grid_h / h_orig)
                
                drawing_state['scale_factor'] = scale
                
                if scale < 1.0:
                        w_new = int(w_orig * scale)
                        h_new = int(h_orig * scale)
                        draw_img = cv2.resize(frame, (w_new, h_new))
                else:
                        draw_img = frame.copy()
                
                pts = drawing_state['points']
                if len(pts) > 0:
                    # Scale points for display
                    display_pts = []
                    for pt in pts:
                        sx = int(pt[0] * scale)
                        sy = int(pt[1] * scale)
                        display_pts.append((sx, sy))
                        
                    if len(display_pts) > 1:
                        cv2.polylines(draw_img, [np.array(display_pts)], False, (0, 255, 0), 2)
                    for pt in display_pts:
                        cv2.circle(draw_img, pt, 4, (0, 0, 255), -1)
                        
                cv2.imshow(win_name, draw_img)

def handle_key_press(key, grid_layout, session_recorder, polygon_manager):
    if key == ord('p'):
        # Toggle Polygon UI - update logic if needed, for now just logs
        # grid_layout['enabled'] logic might need revisit if we use config directly
        # But if we want runtime toggle, we might need mutable state or update config object (in-memory)
        pass 
    elif key == ord('b'):
        # Switch to Belt Mode
        polygon_manager.set_mode('belt')
    elif key == ord('m'):
        # Switch to Mask Mode
        polygon_manager.set_mode('mask')
    elif key == ord('r'):
        
        # Toggle recording
        if session_recorder.get_status():
            session_recorder.stop_session()
            logger.info("Recording STOPPED via UI")
        else:
            session_recorder.start_session()
            logger.info("Recording STARTED via UI")

def ui_worker(queues, polygon_manager):
    last_num_cameras = -1

    # Screen resolution detection
    grid_w, grid_h = [int(resolution * 0.8) for resolution in get_screen_resolution()]
    
    # grid_layout setup
    grid_layout = {
        'cols': 1, 'rows': 1, 
        'cell_w': grid_w, 'cell_h': grid_h,
        'sorted_ips': [],
        'enabled': config.ui.get('enable_polygon_ui', False)
    }
    
    window_name = "Camera Grid"
    cv2.namedWindow(window_name)
    # Use callback from PolygonManager
    cv2.setMouseCallback(window_name, polygon_manager.mouse_callback_main, grid_layout)
    
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    while True:
        # Dynamic UI Config
        grid_layout['enabled'] = config.ui.get('enable_polygon_ui', False)
        
        # Update latest frames from queues
        for ip, q in queues.items():
            try:
                # Get latest available frame
                while not q.empty():
                    timestamp, frame = q.get_nowait()
                    ui_frames[ip] = (timestamp, frame)
            except Empty:
                pass

        # Check for camera count changes and update layout if needed
        num_cameras = len(queues)
        if num_cameras != last_num_cameras:
            last_num_cameras = num_cameras
            if num_cameras == 0:
                time.sleep(0.1)
                continue    
            else:
                update_grid_layout(grid_layout, num_cameras, grid_w, grid_h)
                cell_w = grid_layout['cell_w']
                cell_h = grid_layout['cell_h']
                cols = grid_layout['cols']
        
        
        sorted_ips = sorted(queues.keys())
        grid_layout['sorted_ips'] = sorted_ips
        
        # Draw the grid
        draw_grid_cells(canvas, grid_layout, polygon_manager, grid_w, grid_h)
        
        cv2.imshow(window_name, canvas)
        
        # Handle Drawing Window Logic
        handle_polygon_drawing_window(polygon_manager, grid_w, grid_h)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27: # ESC key
            if polygon_manager.drawing_state['active']:
                polygon_manager.cancel_drawing()
        else: handle_key_press(key, grid_layout, session_recorder, polygon_manager)
        
        time.sleep(0.01)
            
    cv2.destroyAllWindows()
    # Ensure cleanup on exit
    session_recorder.stop_session()

def apply_mask(frame, points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame

if __name__ == "__main__":
    print("Starting the program...")
    
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
    recording_queues = {} # NEW: Dedicated queues for disk writes
    event_queues = {} # NEW: Always-on queue for alarm buffering
    
    polygon_manager = PolygonManager()

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
        t_cap = threading.Thread(target=camera_capture_worker, args=(ip, uq, iq, rq, eq, fps_val), daemon=True)
        t_cap.start()
        
        # Thread 2: Inference (Slow)
        t_inf = threading.Thread(target=inference_worker, args=(ip, iq, model, polygon_manager), daemon=True)
        t_inf.start()

        # Thread 3: Recording (Active only on demand logic inside)
        # t_rec = threading.Thread(target=recording_worker, args=(ip, rq, fps_val), daemon=True)
        # t_rec.start()
        # Replaced with dynamic registration:
        session_recorder.register_camera(ip, rq, fps_val)
        
        # Thread 4: Event Processing (Always Active)
        t_evt = threading.Thread(target=event_worker, args=(ip, eq, fps_val), daemon=True)
        t_evt.start()
        
        print(f"Started decoupled workers for {ip}")

    # Start UI thread
    if config.ui.get('enable_ui', True):
        ui = threading.Thread(target=ui_worker, args=(ui_queues, polygon_manager), daemon=True)
        ui.start()
        print("Started UI thread")
    else:
        print("UI disabled by config")

    # Start Alarm Manager thread
    am_thread = threading.Thread(target=alarm_manager.worker, daemon=True)
    am_thread.start()
    print("Started Alarm Manager thread")
    
    try:
        if config.ui.get('enable_ui', True):
            ui.join()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

