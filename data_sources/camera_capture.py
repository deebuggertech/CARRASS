import time
import cv2
import numpy as np
from typing import Optional
from data_sources.capture import SensorCapture

class CameraCapture(SensorCapture):
    def __init__(self, camera_index: int = 0, recorder=None, player=None,
                 capture_fps: int = 2, image_format: str = 'jpg'):
        super().__init__(recorder, player)
        self.camera_index = camera_index
        self.capture_fps = capture_fps
        self.image_format = image_format
        
        self._frame_delay = 1.0 / capture_fps if capture_fps > 0 else 0.0
        self._last_capture_time = 0.0

        self._cap = None

    def _get_sensor_name(self) -> str:
        return "camera"

    def start(self):
        if self.player:
            super().start()
            return

        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            self._running = False
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

        for _ in range(3):
            _ = self._cap.read()
        super().start()
        
    def stop(self):
        super().stop()

        if self._cap:
            self._cap.release()
            self._cap = None
    
    def _run(self):
        if self.player:
            self._playback()
        else:
            self._capture()

    def _capture(self):
        if not self._cap:
            print("Error: Camera not initialized for live capture")
            return

        try:
            while self._running:
                ret, frame = self._cap.read()

                if not ret:
                    print("Warning: Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp = time.time()

                if self._frame_delay <= 0 or (timestamp - self._last_capture_time) >= self._frame_delay:
                    self._set_data(frame.copy(), timestamp)

                    if self.recorder:
                        self._record_frame(frame, timestamp)
                    
                    self._frame_count += 1
                    self._last_capture_time = timestamp
                else:
                    time.sleep(0.05)

        finally:
            if self._cap:
                self._cap.release()
    
    def _playback(self):
        if not self.player:
            return
        
        try:
            for frame_number, frame_metadata in self.player.playback_realtime(
                sensor_filter=["camera"]
            ):
                if not self._running:
                    break
                
                image_path = self.player.get_data_path(
                    "camera", frame_number, f".{self.image_format}"
                )
                
                if image_path.exists():
                    frame = cv2.imread(str(image_path))
                    if frame is not None:
                        timestamp = frame_metadata.get("timestamp", time.time())
                        self._set_data(frame, timestamp)
                        self._frame_count += 1
                    else:
                        print(f"Warning: Failed to load image: {image_path}")
                else:
                    print(f"Warning: Image file not found: {image_path}")
        
        except Exception as e:
            print(f"Error during playback: {e}")
    
    def _record_frame(self, frame: np.ndarray, timestamp: float):
        if not self.recorder:
            return
        
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1
        
        frame_number = self.recorder.record_frame(
            sensor_name="camera",
            frame_data=frame,
            timestamp=timestamp,
            extra_info={
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "format": self.image_format
            }
        )
        
        image_path = self.recorder.get_data_path(
            "camera", frame_number, f".{self.image_format}"
        )
        cv2.imwrite(str(image_path), frame)
    
    def get_frame(self) -> Optional[np.ndarray]:
        frame, _ = self.get_data()
        return frame
    
    def get_frame_info(self) -> dict:
        frame, timestamp = self.get_data()
        if frame is not None:
            return {
                "shape": frame.shape,
                "dtype": str(frame.dtype),
                "timestamp": timestamp,
                "frame_count": self._frame_count
            }
        return {
            "shape": None,
            "dtype": None,
            "timestamp": 0.0,
            "frame_count": 0
        }
