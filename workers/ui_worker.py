import cv2
import numpy as np
import time
import math
import ctypes
import platform
import logging
from queue import Empty
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

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

def draw_grid_cells(canvas, grid_layout, polygon_manager, alarm_manager, session_recorder, shared_state, ui_frames, grid_w, grid_h):
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
                # Retrieve latest result object from shared_state
                if shared_state is not None and 'detections' in shared_state and ip in shared_state['detections']:
                    res = shared_state['detections'][ip]
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
                if shared_state is not None:
                    if 'capture_fps' in shared_state and ip in shared_state['capture_fps']:
                        fps_text += f" | Cap: {shared_state['capture_fps'][ip]:.1f}"
                    if 'inference_fps' in shared_state and ip in shared_state['inference_fps']:
                        fps_text += f" | Inf: {shared_state['inference_fps'][ip]:.1f}"
                
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
    config = ConfigManager()
    if config.ui.get('enable_polygon_ui', False):
        mode_str = polygon_manager.edit_mode.upper()
        cv2.putText(canvas, f"EDIT MODE: {mode_str}", (grid_w - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)

def handle_polygon_drawing_window(polygon_manager, ui_frames, grid_w, grid_h):
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

def ui_worker(queues, polygon_manager, alarm_manager, session_recorder, shared_state):
    last_num_cameras = -1
    config = ConfigManager()

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
    
    # Internal state for UI frames (consumed from queues)
    ui_frames = {}

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
                # cell_w = grid_layout['cell_w']
                # cell_h = grid_layout['cell_h']
                # cols = grid_layout['cols']
        
        
        sorted_ips = sorted(queues.keys())
        grid_layout['sorted_ips'] = sorted_ips
        
        # Draw the grid
        draw_grid_cells(canvas, grid_layout, polygon_manager, alarm_manager, session_recorder, shared_state, ui_frames, grid_w, grid_h)
        
        cv2.imshow(window_name, canvas)
        
        # Handle Drawing Window Logic
        handle_polygon_drawing_window(polygon_manager, ui_frames, grid_w, grid_h)
        
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

