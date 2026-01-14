import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple
import threading

class DataRecorder:
    def __init__(self, output_dir: str, session_name: Optional[str] = None):
        if session_name is None:
            session_name = f"session_{int(time.time())}"
        
        self.output_dir = Path(output_dir) / session_name
        self.data_dir = self.output_dir / "data"
        self.metadata_file = self.output_dir / "metadata.json"
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._metadata = {
            "session_name": session_name,
            "start_time": time.time(),
            "frames": []
        }
        self._metadata_lock = threading.Lock()
        
        print(f"Recording to: {self.output_dir}")
    
    def record_frame(self, sensor_name: str, frame_data: Any, 
                     timestamp: Optional[float] = None,
                     extra_info: Optional[Dict] = None) -> int:
        if timestamp is None:
            timestamp = time.time()
        
        with self._metadata_lock:
            frame_number = len(self._metadata["frames"])
            
            frame_metadata = {
                "frame_number": frame_number,
                "sensor": sensor_name,
                "timestamp": timestamp,
            }
            
            if extra_info:
                frame_metadata.update(extra_info)
            
            self._metadata["frames"].append(frame_metadata)
            
            return frame_number
    
    def get_data_path(self, sensor_name: str, frame_number: int, extension: str = "") -> Path:
        filename = f"{sensor_name}_{frame_number:06d}{extension}"
        return self.data_dir / filename
    
    def finalize(self):
        with self._metadata_lock:
            self._metadata["end_time"] = time.time()
            duration = self._metadata["end_time"] - self._metadata["start_time"]
            self._metadata["duration"] = duration
            self._metadata["total_frames"] = len(self._metadata["frames"])
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
            
            print(f"Recorded {self._metadata['total_frames']} frames in {duration:.2f} seconds")
            print(f"Metadata saved to: {self.metadata_file}")

class DataPlayer:
    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self.data_dir = self.session_dir / "data"
        self.metadata_file = self.session_dir / "metadata.json"
        
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        with open(self.metadata_file, 'r') as f:
            self._metadata = json.load(f)
        
        self._playback_start_wall_time = None
        self._playback_start_data_time = None
        self._playback_lock = threading.Lock()
        
        print(f"Loaded session: {self._metadata.get('session_name', 'unknown')}")
        print(f"Total frames: {self._metadata.get('total_frames', 0)}")
        print(f"Duration: {self._metadata.get('duration', 0):.2f} seconds")
    
    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()
    
    def get_frame_metadata(self, frame_number: int) -> Optional[Dict[str, Any]]:
        frames = self._metadata.get("frames", [])
        if 0 <= frame_number < len(frames):
            return frames[frame_number].copy()
        return None
    
    def get_data_path(self, sensor_name: str, frame_number: int, extension: str = "") -> Path:
        filename = f"{sensor_name}_{frame_number:06d}{extension}"
        return self.data_dir / filename
    
    def iter_frames(self, sensor_filter: Optional[List[str]] = None) -> Iterator[Tuple[int, Dict[str, Any]]]:
        for frame in self._metadata.get("frames", []):
            if sensor_filter and frame.get("sensor") not in sensor_filter:
                continue
            yield frame["frame_number"], frame
    
    def playback_realtime(self, sensor_filter: Optional[List[str]] = None,
                         speed: float = 1.0) -> Iterator[Tuple[int, Dict[str, Any]]]:
        frame_count = 0
        
        for frame_number, frame_metadata in self.iter_frames(sensor_filter):
            frame_timestamp = frame_metadata.get("timestamp", 0)
            
            with self._playback_lock:
                if self._playback_start_data_time is None:
                    self._playback_start_data_time = frame_timestamp
                    self._playback_start_wall_time = time.perf_counter()
                
                start_data_time = self._playback_start_data_time
                start_wall_time = self._playback_start_wall_time
            
            elapsed_data_time = frame_timestamp - start_data_time
            target_wall_time = start_wall_time + (elapsed_data_time / speed)
            
            while True:
                current_time = time.perf_counter()
                remaining = target_wall_time - current_time
                
                if remaining <= 0:
                    break
                
                sleep_duration = min(remaining * 0.9, 0.005)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            
            frame_count += 1
            yield frame_number, frame_metadata
    
    def get_total_frames(self) -> int:
        return self._metadata.get("total_frames", 0)
    
    def get_duration(self) -> float:
        return self._metadata.get("duration", 0.0)
    
    def get_sensors(self) -> List[str]:
        sensors = set()
        for frame in self._metadata.get("frames", []):
            sensors.add(frame.get("sensor", "unknown"))
        return sorted(list(sensors))
    
    def reset_playback_timing(self):
        with self._playback_lock:
            self._playback_start_wall_time = None
            self._playback_start_data_time = None
