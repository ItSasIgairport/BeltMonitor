import time
import threading
import logging
import cv2
import os
import datetime
from collections import deque
from queue import Queue, Empty, Full
from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)
config = ConfigManager()

class AlarmManager:
    def __init__(self):
        self.alarm_queue = Queue()
        self.alarm_states = {} # { ip: { 'status': 'SAFE'|'WARNING'|'ALARM', 'start_time': timestamp, 'last_seen': timestamp } }
        self.lock = threading.Lock()
        
        # Event Recording State
        self.frame_buffers = {} # ip -> deque of (timestamp, frame)
        self.active_recordings = {} # ip -> { 'writer': cv2.VideoWriter, 'end_time': float, 'file_path': str, 'initialized': bool }
        self.buffer_lock = threading.Lock()

    def add_event(self, ip, has_intrusion, timestamp, info=None):
        """Add an intrusion event to the queue"""
        try:
            self.alarm_queue.put((ip, has_intrusion, timestamp, info), block=False)
        except Full:
            pass # Drop event if queue is full (unlikely)

    def get_status(self, ip):
        """Thread-safe retrieval of alarm status"""
        with self.lock:
            if ip in self.alarm_states:
                return self.alarm_states[ip]['status']
            return 'SAFE'

    def process_frame(self, ip, frame, timestamp=None, fps=30):
        """
        Called by camera thread to buffer frames and handle event recording.
        """
        if timestamp is None:
            timestamp = time.time()
        now = timestamp
        
        # Config fetch (cached or direct)
        event_config = config.config.get('event_recording', {})
        if not event_config.get('enabled', False):
            return

        pre_event_sec = event_config.get('pre_event_sec', 5)
        
        # Get current status
        status = self.get_status(ip)
        
        # --- Dot Color Logic (with Latching) ---
        draw_color = None
        if status == 'WARNING':
            draw_color = (0, 255, 255) # Yellow
        elif status == 'ALARM':
            draw_color = (0, 0, 255) # Red
            
        # Apply Latching logic if we are recording
        rec_state = self.active_recordings.get(ip)
        if rec_state:
            # If current status is ALARM, latch it for this recording
            if status == 'ALARM':
                rec_state['latched_alarm'] = True
            
            # If previously latched, override color to RED
            if rec_state.get('latched_alarm', False):
                draw_color = (0, 0, 255)
        
        frame_to_record = frame.copy()
        if draw_color:
            self._draw_status_dot(frame_to_record, draw_color)

        with self.buffer_lock:
            # 1. Update Buffer
            if ip not in self.frame_buffers:
                self.frame_buffers[ip] = deque()
            
            buffer = self.frame_buffers[ip]
            buffer.append((now, frame_to_record))
            
            # Prune old frames
            while buffer and (now - buffer[0][0] > pre_event_sec):
                buffer.popleft()
                
            # 2. Handle Active Recording
            if ip in self.active_recordings:
                rec_state = self.active_recordings[ip]
                
                if now < rec_state['end_time']:
                    # Initialize writer if needed
                    if not rec_state['initialized']:
                        self._init_writer(ip, frame_to_record, fps, rec_state, now)
                    
                    # Write frame
                    if rec_state['writer']:
                        stats = rec_state.get('sync_stats')
                        if stats:
                            elapsed = now - stats['start_time']
                            expected_frames = int(elapsed * fps)
                            if stats['frames_written'] == 0:
                                expected_frames = 1
                            
                            frames_to_write = expected_frames - stats['frames_written']
                            
                            # SYNC FIX: If frames_to_write < 1, skip writing
                            if frames_to_write > 0:
                                for _ in range(frames_to_write):
                                    rec_state['writer'].write(frame_to_record)
                                    stats['frames_written'] += 1
                        else:
                            # Fallback
                            rec_state['writer'].write(frame_to_record)
                else:
                    # Stop Recording
                    logger.info(f"Event recording finished for {ip}: {rec_state.get('file_path')}")
                    if rec_state['writer']:
                        rec_state['writer'].release()
                    del self.active_recordings[ip]

    def _draw_status_dot(self, frame, color):
        """Draws a colored dot on the frame."""
        # Draw dot in top-right corner
        h, w = frame.shape[:2]
        center = (w - 40, 40) # Slightly more inset due to larger size
        radius = 30 # 3x bigger (was 10)
        
        cv2.circle(frame, center, radius, color, -1)

    def _init_writer(self, ip, frame, fps, rec_state, current_timestamp):
        """Helper to initialize VideoWriter and dump buffer"""
        try:
            h, w = frame.shape[:2]
            save_path = config.config.get('event_recording', {}).get('save_path', 'event_recordings')
            os.makedirs(save_path, exist_ok=True)
            
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"event_{ip.replace('.', '_')}_{timestamp_str}.avi"
            full_path = os.path.join(save_path, filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(full_path, fourcc, fps, (w, h))
            
            if writer.isOpened():
                logger.info(f"Started EVENT recording for {ip}: {full_path}")
                rec_state['writer'] = writer
                rec_state['file_path'] = full_path
                rec_state['initialized'] = True
                
                # Sync stats
                rec_state['sync_stats'] = {'start_time': 0, 'frames_written': 0}
                
                # Dump buffer (pre-event frames)
                buffer = self.frame_buffers[ip]
                frames_to_dump = list(buffer)[:-1] 
                
                # Determine start time from oldest buffered frame or current if empty
                first_frame_time = frames_to_dump[0][0] if frames_to_dump else current_timestamp
                rec_state['sync_stats']['start_time'] = first_frame_time

                for ts, buf_frame in frames_to_dump:
                    stats = rec_state['sync_stats']
                    elapsed = ts - stats['start_time']
                    expected_frames = int(elapsed * fps)
                    
                    if stats['frames_written'] == 0:
                        expected_frames = 1
                        
                    frames_to_write = expected_frames - stats['frames_written']
                    
                    # SYNC FIX: If frames_to_write < 1, skip writing
                    if frames_to_write > 0:
                        for _ in range(frames_to_write):
                            writer.write(buf_frame)
                            stats['frames_written'] += 1

            else:
                logger.error(f"Failed to create event video writer for {full_path}")
                rec_state['writer'] = None 
                rec_state['initialized'] = True 
        except Exception as e:
            logger.error(f"Error initializing event writer for {ip}: {e}")
            rec_state['initialized'] = True 

    def start_event_recording(self, ip, duration):
        """
        Triggers or extends an event recording.
        """
        with self.buffer_lock:
            now = time.time()
            end_time = now + duration
            
            if ip in self.active_recordings:
                # Extend existing recording
                if end_time > self.active_recordings[ip]['end_time']:
                    self.active_recordings[ip]['end_time'] = end_time
                    logger.info(f"Extending event recording for {ip} by {duration}s")
            else:
                # Start new recording (Writer init deferred to process_frame)
                logger.info(f"Triggering EVENT recording for {ip}")
                self.active_recordings[ip] = {
                    'writer': None,
                    'end_time': end_time,
                    'file_path': None,
                    'initialized': False,
                    'latched_alarm': False # Initialize latch state
                }

    def trigger_emergency_pipeline(self, ip):
        """Stage 3: Emergency Pipeline Trigger"""
        logger.error(f"!!! EMERGENCY PIPELINE TRIGGERED FOR {ip} !!!")
        # Placeholder for actual emergency logic (GPIO, API, Email, etc.)
        # Example: session_recorder.trigger_event(ip) (if supported)

    def worker(self):
        logger.info("Alarm Manager thread started")
        
        while True:
            trigger_time = config.alarm.get('trigger_time', 1.0)
            event_config = config.config.get('event_recording', {})
            
            try:
                # Process all available events in the queue
                while not self.alarm_queue.empty():
                    ip, intrusion, timestamp, info = self.alarm_queue.get()
                    
                    with self.lock:
                        if ip not in self.alarm_states:
                            self.alarm_states[ip] = { 'status': 'SAFE', 'start_time': 0, 'last_seen': 0 }
                        
                        state = self.alarm_states[ip]
                        
                        # Check cooldown
                        if timestamp < state.get('cooldown_expiry', 0):
                             intrusion = False 

                    if intrusion:
                        state['last_seen'] = timestamp
                        if state['status'] == 'SAFE':
                            state['status'] = 'WARNING'
                            state['start_time'] = timestamp
                            
                            # Construct Log Message
                            log_msg = f"{ip} - Status changed to WARNING"
                            if info:
                                log_msg += f" | Info: {info}"
                            logger.info(log_msg)
                            
                            # Trigger Recording on WARNING
                            if event_config.get('enabled', False) and event_config.get('trigger_on_warning', False):
                                post_time = event_config.get('post_event_sec', 10)
                                self.start_event_recording(ip, post_time)

                    else:
                         # Immediate reset on clear
                         if state['status'] != 'SAFE':
                            state['status'] = 'SAFE'
                            state['start_time'] = 0
                            logger.info(f"{ip} - Status reset to SAFE")
                
                # Check time-based transitions
                now = time.time()
                with self.lock:
                    for ip, state in self.alarm_states.items():
                        if state['status'] == 'WARNING':
                            duration = now - state['start_time']
                            if duration >= trigger_time:
                                state['status'] = 'ALARM'
                                logger.warning(f"{ip} - ALARM ACTIVATED! Duration: {duration:.2f}s")
                                self.trigger_emergency_pipeline(ip)
                                
                                # Trigger Recording on ALARM
                                if event_config.get('enabled', False) and event_config.get('trigger_on_alarm', False):
                                    post_time = event_config.get('post_event_sec', 10)
                                    self.start_event_recording(ip, post_time)
                                    
                                    # Set Cooldown
                                    state['cooldown_expiry'] = now + post_time
                                    logger.info(f"{ip} - Cooldown activated until {state['cooldown_expiry']}")

                time.sleep(0.01) # Prevent busy loop
            except Exception as e:
                logger.error(f"Error in Alarm Manager: {e}")
                time.sleep(1)
