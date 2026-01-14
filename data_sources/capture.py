import threading
import time
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

class SensorCapture(ABC):
    def __init__(self, recorder=None, player=None):
        self.recorder = recorder
        self.player = player
        
        self._data = None
        self._data_timestamp = 0.0
        self._data_lock = threading.Lock()

        self._running = False
        self._worker_thread = None

        self._frame_count = 0

    @abstractmethod
    def _run(self):
        pass
    
    @abstractmethod
    def _get_sensor_name(self) -> str:
        pass
    
    def _set_data(self, data: Any, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        
        with self._data_lock:
            self._data = data
            self._data_timestamp = timestamp

    def start(self):
        if self._running:
            print(f"Already running: {self._get_sensor_name()}")
            return
        
        self._running = True
        self._frame_count = 0
        self._worker_thread = threading.Thread(target=self._run, daemon=True)
        self._worker_thread.start()
        print(f"Started: {self._get_sensor_name()}")

    def stop(self):
        if not self._running:
            return
        
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        if self.recorder:
            self.recorder.finalize()

        print(f"Stopped: {self._get_sensor_name()}")

    def get_data(self) -> Tuple[Any, float]:
        with self._data_lock:
            return (self._data, self._data_timestamp)

    def is_running(self) -> bool:
        return self._running
    
    def get_frame_count(self) -> int:
        return self._frame_count
