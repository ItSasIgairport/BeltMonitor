import json
import os
import logging
import threading
import cv2

logger = logging.getLogger(__name__)

class PolygonManager:
    def __init__(self, filepath="polygons.json"):
        self.filepath = filepath
        self.polygons = {}  # { ip: { "points": [[x,y], ...], "resolution": [w, h] } }
        self.lock = threading.Lock()
        self.mouse_pos = (0, 0)
        
        # Drawing state
        self.drawing_state = {
            'active': False,
            'ip': None,
            'points': [], # List of (x,y) - ORIGINAL resolution coordinates
            'window_name': None,
            'frame_shape': None, # (h, w) - ORIGINAL resolution
            'window_initialized': False,
            'scale_factor': 1.0 # scale = displayed / original
        }
        
        self.load()

    def load(self):
        with self.lock:
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, 'r') as f:
                        data = json.load(f)
                        # Validate structure lightly if needed
                        self.polygons = data
                        logger.info(f"Loaded polygons from {self.filepath}")
                except Exception as e:
                    logger.error(f"Failed to load polygons: {e}")
                    self.polygons = {}
            else:
                self.polygons = {}

    def save(self):
        with self.lock:
            try:
                with open(self.filepath, 'w') as f:
                    json.dump(self.polygons, f, indent=2)
                logger.info(f"Saved polygons to {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to save polygons: {e}")

    def save_polygon(self, ip, points, resolution):
        """
        Save a polygon for a specific IP.
        points: List of (x, y) tuples or lists.
        resolution: tuple or list (width, height) of the source frame.
        """
        with self.lock:
            self.polygons[ip] = {
                "points": points,
                "resolution": resolution
            }
        self.save()

    def get_polygon(self, ip):
        """
        Returns the polygon dict for an IP or None.
        return: { "points": [...], "resolution": [...] }
        """
        with self.lock:
            return self.polygons.get(ip)

    def delete_polygon(self, ip):
        with self.lock:
            if ip in self.polygons:
                del self.polygons[ip]
                self.save()

    def mouse_callback_main(self, event, x, y, flags, param):
        """
        Callback for the main grid window.
        param: grid_layout dictionary containing 'cols', 'rows', 'cell_w', 'cell_h', 'sorted_ips', 'enabled'
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)

        if not param.get('enabled', True):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Determine which cell was clicked
            cols = param.get('cols', 1)
            cell_w = param.get('cell_w', 0)
            cell_h = param.get('cell_h', 0)
            sorted_ips = param.get('sorted_ips', [])
            
            if cell_w == 0 or cell_h == 0: return

            col = x // cell_w
            row = y // cell_h
            index = row * cols + col
            
            if 0 <= index < len(sorted_ips):
                ip = sorted_ips[index]
                # Start drawing mode
                self.drawing_state['active'] = True
                self.drawing_state['ip'] = ip
                self.drawing_state['points'] = []
                self.drawing_state['window_name'] = f"Edit Polygon - {ip}"
                self.drawing_state['window_initialized'] = False
                self.drawing_state['scale_factor'] = 1.0 # Reset scale factor
                logger.info(f"Started drawing for {ip}")

    def mouse_callback_draw(self, event, x, y, flags, param):
        """
        Callback for the dedicated drawing window.
        """
        if not self.drawing_state['active']:
            return
            
        scale = self.drawing_state.get('scale_factor', 1.0)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates (x, y) back to original resolution
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            self.drawing_state['points'].append((orig_x, orig_y))
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Finish and save
            ip = self.drawing_state['ip']
            points = self.drawing_state['points']
            # Points are already in original resolution
            if self.drawing_state['frame_shape'] and len(points) > 2:
                h, w = self.drawing_state['frame_shape'][:2]
                self.save_polygon(ip, points, (w, h))
                logger.info(f"Saved polygon for {ip} with {len(points)} points")
            
            # Close window
            self.drawing_state['active'] = False
            if self.drawing_state['window_name']:
                cv2.destroyWindow(self.drawing_state['window_name'])
