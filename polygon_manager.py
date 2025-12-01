import json
import os
import logging
import threading
import cv2

logger = logging.getLogger(__name__)

class PolygonManager:
    def __init__(self, filepath="polygons.json"):
        self.filepath = filepath
        self.polygons = {}  # { ip: { "belt": {...}, "mask": {...} } }
        self.lock = threading.Lock()
        self.edit_mode = 'belt' # 'belt' or 'mask'
        
        # Drawing state
        self.drawing_state = {
            'active': False,
            'ip': None,
            'points': [], # List of (x,y) - ORIGINAL resolution coordinates
            'window_name': None,
            'frame_shape': None, # (h, w) - ORIGINAL resolution
            'window_initialized': False,
            'scale_factor': 1.0, # scale = displayed / original
            'type': 'belt'
        }
        
        self.load()

    def set_mode(self, mode):
        if mode in ['belt', 'mask']:
            self.edit_mode = mode
            logger.info(f"Polygon edit mode set to: {mode}")

    def load(self):
        with self.lock:
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, 'r') as f:
                        data = json.load(f)
                        # Migration check: if keys are missing "belt"/"mask" and have "points" directly
                        migrated = {}
                        for ip, val in data.items():
                            if "points" in val:
                                # Legacy format, assume it is 'belt'
                                migrated[ip] = {
                                    "belt": val,
                                    "mask": None
                                }
                            else:
                                migrated[ip] = val
                        
                        self.polygons = migrated
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

    def save_polygon(self, ip, points, resolution, poly_type="belt"):
        """
        Save a polygon for a specific IP and type.
        points: List of (x, y) tuples or lists.
        resolution: tuple or list (width, height) of the source frame.
        poly_type: 'belt' or 'mask'
        """
        with self.lock:
            if ip not in self.polygons:
                self.polygons[ip] = {}
            
            self.polygons[ip][poly_type] = {
                "points": points,
                "resolution": resolution
            }
        self.save()

    def get_polygon(self, ip, poly_type="belt"):
        """
        Returns the polygon dict for an IP and type or None.
        return: { "points": [...], "resolution": [...] }
        """
        with self.lock:
            if ip in self.polygons:
                return self.polygons[ip].get(poly_type)
            return None

    def delete_polygon(self, ip, poly_type="belt"):
        with self.lock:
            if ip in self.polygons and poly_type in self.polygons[ip]:
                del self.polygons[ip][poly_type]
                self.save()

    def mouse_callback_main(self, event, x, y, flags, param):
        """
        Callback for the main grid window.
        param: grid_layout dictionary containing 'cols', 'rows', 'cell_w', 'cell_h', 'sorted_ips', 'enabled'
        """

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
                self.drawing_state['type'] = self.edit_mode
                self.drawing_state['window_name'] = f"Edit {self.edit_mode.upper()} - {ip}"
                self.drawing_state['window_initialized'] = False
                self.drawing_state['scale_factor'] = 1.0 # Reset scale factor
                logger.info(f"Started drawing {self.edit_mode} for {ip}")

    def cancel_drawing(self):
        """
        Cancels the current drawing operation.
        """
        if self.drawing_state['active']:
            self.drawing_state['active'] = False
            if self.drawing_state['window_name']:
                cv2.destroyWindow(self.drawing_state['window_name'])
            logger.info("Drawing cancelled by user")

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
            poly_type = self.drawing_state.get('type', 'belt')
            
            # Points are already in original resolution
            if self.drawing_state['frame_shape'] and len(points) > 2:
                h, w = self.drawing_state['frame_shape'][:2]
                self.save_polygon(ip, points, (w, h), poly_type)
                logger.info(f"Saved {poly_type} polygon for {ip} with {len(points)} points")
            
            # Close window
            self.drawing_state['active'] = False
            if self.drawing_state['window_name']:
                cv2.destroyWindow(self.drawing_state['window_name'])
