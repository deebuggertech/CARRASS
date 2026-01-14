import time
import json
from typing import Optional, Tuple
from data_sources.capture import SensorCapture

class RadarCapture(SensorCapture):
    FRAME_HEADER_SIZE = 2
    PAYLOAD_SIZE = 28
    MINIMUM_FRAME_SIZE = 30
    HEADER_BYTE_1 = b'\xAA'
    HEADER_BYTE_2 = b'\xFF'
    
    SIGNED_OFFSET = 32768
    METERS_SCALE = 1000.0
    
    def __init__(self, port: str = 'COM8', baud_rate: int = 256000,
                 recorder=None, player=None, capture_fps: int = 2):
        super().__init__(recorder, player)
        self.port = port
        self.baud_rate = baud_rate

        self.capture_fps = capture_fps
        self._frame_delay = 1.0 / capture_fps if capture_fps > 0 else 0.0
        self._last_capture_time = 0.0

    def _get_sensor_name(self) -> str:
        return "radar"
    
    def _run(self):
        if self.player:
            self._playback()
        else:
            self._capture()

    def _capture(self):
        try:
            import serial
        except ImportError:
            print("Error: pyserial not installed. Install with: pip install pyserial")
            self._running = False
            return
        
        try:
            ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            print(f"Connected to radar on {self.port} at {self.baud_rate} baud")
        except serial.SerialException as e:
            print(f"Error: Could not open serial port {self.port}: {e}")
            self._running = False
            return
        
        try:
            while self._running:
                if ser.in_waiting >= self.MINIMUM_FRAME_SIZE:
                    byte1 = ser.read(1)
                    if byte1 == self.HEADER_BYTE_1:
                        byte2 = ser.read(1)
                        if byte2 == self.HEADER_BYTE_2:
                            payload = ser.read(self.PAYLOAD_SIZE)
                            
                            if len(payload) == self.PAYLOAD_SIZE:
                                raw_x = int.from_bytes(payload[2:4], byteorder='little', signed=True)
                                raw_y = int.from_bytes(payload[4:6], byteorder='little', signed=True)
                                
                                x_m = (raw_x % self.SIGNED_OFFSET if raw_x > 0 else -(raw_x % self.SIGNED_OFFSET)) / self.METERS_SCALE
                                x_m = -x_m
                                y_m = (raw_y % self.SIGNED_OFFSET) / self.METERS_SCALE
                                
                                timestamp = time.time()

                                if self._frame_delay <= 0 or (timestamp - self._last_capture_time) >= self._frame_delay:
                                    radar_data = {
                                        "x": x_m,
                                        "y": y_m,
                                        "raw_x": raw_x,
                                        "raw_y": raw_y
                                    }
                                    self._set_data(radar_data, timestamp)

                                    if self.recorder:
                                        self._record_data(radar_data, timestamp)

                                    self._frame_count += 1
                                    self._last_capture_time = timestamp
                else:
                    time.sleep(0.05)
        
        except Exception as e:
            print(f"Radar capture error: {e}")
        
        finally:
            ser.close()
            print("Radar serial port closed")
    
    def _playback(self):
        if not self.player:
            return
        
        try:
            for frame_number, frame_metadata in self.player.playback_realtime(
                sensor_filter=["radar"]
            ):
                if not self._running:
                    break
                
                radar_data = {
                    "x": frame_metadata.get("x", 0.0),
                    "y": frame_metadata.get("y", 0.0),
                    "raw_x": frame_metadata.get("raw_x", 0),
                    "raw_y": frame_metadata.get("raw_y", 0)
                }
                
                timestamp = frame_metadata.get("timestamp", time.time())
                self._set_data(radar_data, timestamp)
                self._frame_count += 1

        except Exception as e:
            print(f"Error during playback: {e}")
    
    def _record_data(self, radar_data: dict, timestamp: float):
        if not self.recorder:
            return
        
        self.recorder.record_frame(
            sensor_name="radar",
            frame_data=radar_data,
            timestamp=timestamp,
            extra_info=radar_data
        )
    
    def get_position(self) -> Optional[Tuple[float, float]]:
        data, _ = self.get_data()
        if data:
            return (data.get("x", 0.0), data.get("y", 0.0))
        return None
    
    def get_radar_data(self) -> Optional[dict]:
        data, _ = self.get_data()
        return data
    
    def get_radar_info(self) -> dict:
        data, timestamp = self.get_data()
        if data:
            return {
                "x": data.get("x", 0.0),
                "y": data.get("y", 0.0),
                "raw_x": data.get("raw_x", 0),
                "raw_y": data.get("raw_y", 0),
                "timestamp": timestamp,
                "frame_count": self._frame_count
            }
        return {
            "x": 0.0,
            "y": 0.0,
            "raw_x": 0,
            "raw_y": 0,
            "timestamp": 0.0,
            "frame_count": 0
        }
