import cv2
import time
import logging
from alarm_manager import AlarmManager

logger = logging.getLogger(__name__)

def event_worker(ip, event_queue, fps, alarm_manager):
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
             time.sleep(0.1)

