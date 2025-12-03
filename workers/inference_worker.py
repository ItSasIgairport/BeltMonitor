import cv2
import numpy as np
import logging
import time
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

def apply_mask(frame, points):
    """
    Applies a polygon mask to the frame.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame

def inference_worker(ip, inference_queue, model, polygon_manager, alarm_manager, shared_state):
    """
    Slow worker: Runs YOLO inference on latest available frame.
    Updates shared state 'detections' with raw results.
    """
    config = ConfigManager()

    # FPS Tracking
    fps_start_time = time.time()
    fps_frame_count = 0

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
                t0 = time.time()
                results = model.predict(curr_frame, verbose=False, conf=conf_threshold)
                t1 = time.time()
                inf_time_ms = (t1 - t0) * 1000
                if shared_state is not None and 'inference_time' in shared_state:
                    shared_state['inference_time'][ip] = inf_time_ms
            
            # Retrieve belt polygon for logic check
            belt_poly_data = polygon_manager.get_polygon(ip, 'belt')

            current_intrusion = False
            detection_info = None

            # Store RAW results for UI to draw later
            if results:
                res = results[0]
                if shared_state is not None and 'detections' in shared_state:
                    shared_state['detections'][ip] = res # Store the object
                
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
                                if kp_idx < len(person_kpts):
                                    x, y = person_kpts[kp_idx]
                                    # Handle potential index error if kpts_conf is structured differently or check length
                                    conf = 1.0
                                    if kpts_conf is not None and i < len(kpts_conf) and kp_idx < len(kpts_conf[i]):
                                         conf = kpts_conf[i][kp_idx]

                                    if conf > conf_threshold and (x > 0 or y > 0):
                                        if cv2.pointPolygonTest(belt_points, (float(x), float(y)), False) >= 0:
                                            person_on_belt = True
                                            detection_info = {'confidence': float(conf), 'keypoint_index': kp_idx}
                                            break
                            
                            if person_on_belt:
                                current_intrusion = True
                                break 

            # --- Push Event to Alarm Manager ---
            alarm_manager.add_event(ip, current_intrusion, timestamp, detection_info)

            # FPS Calculation
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                curr_fps = fps_frame_count / elapsed
                if shared_state is not None and 'inference_fps' in shared_state:
                    shared_state['inference_fps'][ip] = curr_fps
                fps_frame_count = 0
                fps_start_time = time.time()

        except Exception as e:
            logger.error(f"Error in inference logic for {ip}: {e}")
            time.sleep(0.1) # Prevent tight loop on error
