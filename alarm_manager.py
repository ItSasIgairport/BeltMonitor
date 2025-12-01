import time
import threading
import logging
from queue import Queue, Empty, Full
from config_manager import ConfigManager

logger = logging.getLogger(__name__)
config = ConfigManager()

class AlarmManager:
    def __init__(self):
        self.alarm_queue = Queue()
        self.alarm_states = {} # { ip: { 'status': 'SAFE'|'WARNING'|'ALARM', 'start_time': timestamp, 'last_seen': timestamp } }
        self.lock = threading.Lock()

    def add_event(self, ip, has_intrusion, timestamp):
        """Add an intrusion event to the queue"""
        try:
            self.alarm_queue.put((ip, has_intrusion, timestamp), block=False)
        except Full:
            pass # Drop event if queue is full (unlikely)

    def get_status(self, ip):
        """Thread-safe retrieval of alarm status"""
        with self.lock:
            if ip in self.alarm_states:
                return self.alarm_states[ip]['status']
            return 'SAFE'

    def trigger_emergency_pipeline(self, ip):
        """Stage 3: Emergency Pipeline Trigger"""
        logger.error(f"!!! EMERGENCY PIPELINE TRIGGERED FOR {ip} !!!")
        # Placeholder for actual emergency logic (GPIO, API, Email, etc.)
        # Example: session_recorder.trigger_event(ip) (if supported)

    def worker(self):
        logger.info("Alarm Manager thread started")
        trigger_time = config.alarm.get('trigger_time', 1.0)
        
        while True:
            try:
                # Process all available events in the queue
                while not self.alarm_queue.empty():
                    ip, intrusion, timestamp = self.alarm_queue.get()
                    
                    with self.lock:
                        if ip not in self.alarm_states:
                            self.alarm_states[ip] = { 'status': 'SAFE', 'start_time': 0, 'last_seen': 0 }
                        
                        state = self.alarm_states[ip]
                        
                        if intrusion:
                            state['last_seen'] = timestamp
                            if state['status'] == 'SAFE':
                                state['status'] = 'WARNING'
                                state['start_time'] = timestamp
                                logger.info(f"{ip} - Status changed to WARNING")
                        else:
                             # Immediate reset on clear (can add debouncing here if needed)
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
            
                time.sleep(0.01) # Prevent busy loop
            except Exception as e:
                logger.error(f"Error in Alarm Manager: {e}")
                time.sleep(1)
