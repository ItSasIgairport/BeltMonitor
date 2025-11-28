import cv2
import numpy as np
import torch
from ultralytics import YOLO
# from multiprocessing import Process, Queue
from collections import deque
from recorder import record_segment_from_frames, SizeLimitedFileHandler, SessionRecorder
from polygon_manager import PolygonManager
import time
import logging
import os
import threading
import platform
import subprocess
import math
import ctypes
from queue import Queue, Empty, Full

USERNAME = "root"
PASSWORD = "wPro@3!4Gen*"
SCALE_PERCENTAGE = 100
FPS = 30
ENABLE_POLYGON_UI = False # Disabled by default, toggled with 'p'

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

                canvas[y1:y2, x1:x2] = resized
                
                # Draw Saved Polygon
                poly_data = polygon_manager.get_polygon(ip)
                if poly_data:
                    orig_points = poly_data['points']
                    orig_w, orig_h = poly_data['resolution']
                    
                    scale_x = w_curr / orig_w
                    scale_y = h_curr / orig_h
                    
                    scaled_points = []
                    for pt in orig_points:
                        sx = int(pt[0] * scale_x) + x1
                        sy = int(pt[1] * scale_y) + y1
                        scaled_points.append([sx, sy])
                        
                    if len(scaled_points) > 0:
                        poly_cnt = np.array(scaled_points)
                        
                        # Collision Detection Test
                        mx, my = polygon_manager.mouse_pos
                        rect_half = 10
                        collision = False
                        
                        # Check 4 corners of the mouse rectangle
                        corners = [
                            (mx - rect_half, my - rect_half),
                            (mx + rect_half, my - rect_half),
                            (mx + rect_half, my + rect_half),
                            (mx - rect_half, my + rect_half)
                        ]
                        
                        for cx, cy in corners:
                            if cv2.pointPolygonTest(poly_cnt, (cx, cy), False) >= 0:
                                collision = True
                                break
                        
                        color = (0, 0, 255) if collision else (0, 255, 0)
                        cv2.polylines(canvas, [poly_cnt], True, color, 2)
                
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
    
    # Edit Mode Indicator
    if grid_layout['enabled']:
            cv2.putText(canvas, "EDIT MODE", (grid_w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)

    # Draw Mouse Cursor Rect for Collision Test
    mx, my = polygon_manager.mouse_pos
    cv2.rectangle(canvas, (mx - 10, my - 10), (mx + 10, my + 10), (255, 255, 0), 1)

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

def handle_key_press(key, grid_layout, session_recorder):
    if key == ord('p'):
        # Toggle Polygon UI
        grid_layout['enabled'] = not grid_layout['enabled']
        state = "ENABLED" if grid_layout['enabled'] else "DISABLED"
        logger.info(f"Polygon UI {state}")
    elif key == ord('r'):
        # Toggle recording
        if session_recorder.get_status():
            session_recorder.stop_session()
            logger.info("Recording STOPPED via UI")
        else:
            session_recorder.start_session()
            logger.info("Recording STARTED via UI")

def ui_worker(queues):
    last_num_cameras = -1

    # Screen resolution detection
    grid_w, grid_h = [int(resolution * 0.8) for resolution in get_screen_resolution()]
    
    
    polygon_manager = PolygonManager()
    
    # State for main grid layout, shared with callback
    grid_layout = {
        'cols': 1, 'rows': 1, 
        'cell_w': grid_w, 'cell_h': grid_h,
        'sorted_ips': [],
        'enabled': ENABLE_POLYGON_UI
    }
    
    
    window_name = "Camera Grid"
    cv2.namedWindow(window_name)
    # Use callback from PolygonManager
    cv2.setMouseCallback(window_name, polygon_manager.mouse_callback_main, grid_layout)
    
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
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
        else: handle_key_press(key, grid_layout, session_recorder)
        
        
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
