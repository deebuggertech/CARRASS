import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import math
import cv2
from utils.config import CAMERA_FOV_DEGREE


class KalmanFilter2D:
    def __init__(
        self,
        process_variance_range: float = 0.05,
        process_variance_azimuth: float = 0.05,
        radar_measurement_variance_range: float = 0.01,
        radar_measurement_variance_azimuth: float = 0.05,
        camera_measurement_variance: float = 0.1,
        initial_covariance: float = 1.0,
        verbose: bool = False
    ):
        self.Q = np.diag([process_variance_range, process_variance_azimuth])
        self.R_radar = np.diag([radar_measurement_variance_range, radar_measurement_variance_azimuth])
        self.R_camera = camera_measurement_variance
        self.initial_covariance = initial_covariance
        self.verbose = verbose

        self.state = np.zeros(2)
        self.P = np.eye(2) * initial_covariance
        self.initialized = False
    
    def predict(self):
        self.P = self.P + self.Q

    def update_radar(self, measurement: np.ndarray):
        if not self.initialized:
            self.state = measurement
            self.initialized = True
            return
        
        H = np.eye(2)
        innovation = measurement - self.state
        S = H @ self.P @ H.T + self.R_radar
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ innovation
        self.P = (np.eye(2) - K @ H) @ self.P

    def update_camera(self, measurement: float):
        if not self.initialized:
            return
        
        H = np.array([[0, 1]])
        innovation = measurement - self.state[1]
        S = H @ self.P @ H.T + self.R_camera
        K = self.P @ H.T / S
        self.state = self.state + K.flatten() * innovation
        self.P = (np.eye(2) - K @ H) @ self.P

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.state.copy(), self.P.copy()
    
    def reset(self):
        self.state = np.zeros(2)
        self.P = np.eye(2) * self.initial_covariance
        self.initialized = False


def pixel_to_azimuth(
    center_x: float,
    camera_width_px: int,
    camera_fov_rad: float = CAMERA_FOV_DEGREE * (math.pi / 180.0)
) -> float:
    normalized_x = (center_x / camera_width_px) - 0.5
    azimuth_angle = normalized_x * camera_fov_rad
    return azimuth_angle


def get_dynamic_association_threshold(distance: float) -> float:
    threshold_at_0m = 50.0
    threshold_at_5m = 20.0
    distance_max = 5.0
    if distance <= 0:
        threshold_deg = threshold_at_0m
    elif distance >= distance_max:
        threshold_deg = threshold_at_5m
    else:
        threshold_deg = threshold_at_0m + (threshold_at_5m - threshold_at_0m) * (distance / distance_max)
    return math.radians(threshold_deg)


def merge_close_contours(
    contour_data: List[Dict],
    merge_distance_px: float
) -> List[Dict]:
    if not contour_data or len(contour_data) <= 1:
        return contour_data

    merged = []
    used = [False] * len(contour_data)
    
    for i in range(len(contour_data)):
        if used[i]:
            continue
        
        group = [contour_data[i]]
        used[i] = True
        
        for j in range(i + 1, len(contour_data)):
            if used[j]:
                continue
            
            x_dist = abs(contour_data[i]['center_of_mass_x'] - contour_data[j]['center_of_mass_x'])
            if x_dist < merge_distance_px:
                group.append(contour_data[j])
                used[j] = True
        
        if len(group) == 1:
            merged.append(group[0])
        else:
            all_points = []
            for item in group:
                contour = item['contour']
                all_points.extend(contour.reshape(-1, 2).tolist())
            
            all_points = np.array(all_points)
            merged_item = {
                'contour': all_points,
                'center_of_mass_x': np.mean(all_points[:, 0]),
                'covariance_x': np.var(all_points[:, 0])
            }
            merged.append(merged_item)

    return merged


def track_objects(
    radar_data: Optional[dict],
    contour_data: Optional[List[Dict]],
    trackers: List[KalmanFilter2D],
    camera_width_px: int,
    original_frame: Optional[np.ndarray] = None,
    merge_distance_px: float = 250.0,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    camera_fov_rad = math.radians(CAMERA_FOV_DEGREE)
    
    if contour_data:
        contour_data = merge_close_contours(contour_data, merge_distance_px)

    objects = []
    
    if not radar_data or 'x' not in radar_data or 'y' not in radar_data:
        radar_detections = []
    else:
        x_radar = float(radar_data['x'])
        y_radar = float(radar_data['y'])

        if y_radar > 0:
            range_val = math.sqrt(x_radar**2 + y_radar**2)
            azimuth_angle = math.atan2(x_radar, y_radar)
            radar_detections = [{'range': range_val, 'azimuth': azimuth_angle, 'x': x_radar, 'y': y_radar}]
        else:
            radar_detections = []

    if not radar_detections and not contour_data:
        for i, tracker in enumerate(trackers):
            tracker.predict()
            if tracker.initialized:
                state, P = tracker.get_state()
                range_val, azimuth = state
                x = range_val * math.sin(azimuth)
                y = range_val * math.cos(azimuth)
                obj = {
                    'range': float(range_val),
                    'azimuth': float(azimuth),
                    'x': float(x),
                    'y': float(y),
                    'contour': None
                }
                objects.append(obj)
        return objects
    
    if len(trackers) == 0 and radar_detections:
        tracker = KalmanFilter2D(verbose=verbose)
        trackers.append(tracker)
    
    for tracker_idx, tracker in enumerate(trackers):
        tracker.predict()

        if radar_detections:
            detection = radar_detections[0]
            measurement = np.array([detection['range'], detection['azimuth']])
            tracker.update_radar(measurement)

        if contour_data and tracker.initialized:
            state, _ = tracker.get_state()
            range_val, azimuth = state

            dynamic_threshold_rad = get_dynamic_association_threshold(range_val)

            best_contour = None
            best_distance = float('inf')

            for i, contour in enumerate(contour_data):
                center_x = contour['center_of_mass_x']
                azimuth_camera = pixel_to_azimuth(center_x, camera_width_px, camera_fov_rad)
                
                distance = abs(azimuth_camera - azimuth)

                if distance < best_distance and distance < dynamic_threshold_rad:
                    best_distance = distance
                    best_contour = contour
            
            if best_contour:
                center_x = best_contour['center_of_mass_x']
                azimuth_camera = pixel_to_azimuth(center_x, camera_width_px, camera_fov_rad)
                tracker.update_camera(azimuth_camera)

        if tracker.initialized:
            state, P = tracker.get_state()
            range_val, azimuth = state
            x = range_val * math.sin(azimuth)
            y = range_val * math.cos(azimuth)

            obj = {
                'range': float(range_val),
                'azimuth': float(azimuth),
                'x': float(x),
                'y': float(y),
                'contour': None
            }
            
            dynamic_threshold_rad = get_dynamic_association_threshold(range_val)

            if contour_data:
                for contour in contour_data:
                    center_x = contour['center_of_mass_x']
                    azimuth_camera = pixel_to_azimuth(center_x, camera_width_px, camera_fov_rad)
                    
                    if abs(azimuth_camera - azimuth) < dynamic_threshold_rad:
                        obj['contour'] = contour
                        break

            objects.append(obj)

    if verbose:
        visualize_kalman(radar_data, contour_data, objects, camera_width_px, original_frame)

    return objects


def visualize_kalman(
    radar_data: Optional[dict],
    contour_data: Optional[List[Dict]],
    tracked_objects: List[Dict[str, Any]],
    camera_width_px: int,
    original_frame: Optional[np.ndarray] = None,
    debug_frame_w: int = 1000,
    debug_frame_h: int = 800,
    win: str = 'Kalman Debug'
) -> None:
    img = np.zeros((debug_frame_h, debug_frame_w, 3), dtype=np.uint8)
    R_MAX = 4.0
    camera_fov_rad = math.radians(CAMERA_FOV_DEGREE)

    margin = 20
    R_px = min(debug_frame_w, debug_frame_h) // 2 - margin
    cx, cy = debug_frame_w // 2, debug_frame_h
    cv2.ellipse(img, (cx, cy), (R_px, R_px), 0, 180, 360, (30, 30, 30), 2)
    
    for user_angle in (CAMERA_FOV_DEGREE / 2, -CAMERA_FOV_DEGREE / 2):
        std_deg = user_angle + 90.0
        theta_rad = math.radians(std_deg)
        end_x = cx + int(round(R_px * math.cos(theta_rad)))
        end_y = cy - int(round(R_px * math.sin(theta_rad)))
        cv2.line(img, (cx, cy), (end_x, end_y), (200, 200, 200), 2)
        cv2.circle(img, (end_x, end_y), 4, (200, 200, 200), -1)
    
    if radar_data and 'x' in radar_data and 'y' in radar_data:
        x = float(radar_data['x'])
        y = float(radar_data['y'])
        if y >= 0:
            r = math.hypot(x, y)
            theta = math.atan2(y, x)
            
            r_plot = min(r, R_MAX)
            radius_px = int(round((r_plot / R_MAX) * R_px))
            
            px = cx + int(round(radius_px * math.cos(theta)))
            py = cy - int(round(radius_px * math.sin(theta)))
            
            px = max(0, min(debug_frame_w - 1, px))
            py = max(0, min(debug_frame_h - 1, py))
            
            cv2.circle(img, (px, py), 8, (255, 255, 0), -1)
            cv2.putText(img, "Radar", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if contour_data:
        for i, contour in enumerate(contour_data):
            center_x = contour['center_of_mass_x']
            azimuth_camera = pixel_to_azimuth(center_x, camera_width_px, camera_fov_rad)

            theta = math.pi / 2 - azimuth_camera

            start_point = (cx, cy)

            end_px = cx + int(round(R_px * math.cos(theta)))
            end_py = cy - int(round(R_px * math.sin(theta)))

            end_px = max(0, min(debug_frame_w - 1, end_px))
            end_py = max(0, min(debug_frame_h - 1, end_py))

            cv2.line(img, start_point, (end_px, end_py), (255, 0, 255), 1, cv2.LINE_AA)

            if original_frame is not None:
                contour_points = contour['contour']
                x, y, w, h = cv2.boundingRect(contour_points)

                pad = 50
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(original_frame.shape[1], x + w + pad)
                y2 = min(original_frame.shape[0], y + h + pad)

                if x2 > x1 and y2 > y1:
                    crop = original_frame[y1:y2, x1:x2]

                    thumb_size = 150
                    crop_h, crop_w = crop.shape[:2]
                    scale = min(thumb_size / crop_h, thumb_size / crop_w)
                    thumb_w = int(crop_w * scale)
                    thumb_h = int(crop_h * scale)
                    thumb = cv2.resize(crop, (thumb_w, thumb_h))

                    thumb_y = max(0, end_py - thumb_h - 25)
                    thumb_x = max(0, end_px - thumb_w // 2)

                    thumb_x = min(thumb_x, debug_frame_w - thumb_w)
                    thumb_y = min(thumb_y, debug_frame_h - thumb_h)

                    try:
                        if len(thumb.shape) == 2:
                            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                        else:
                            thumb_bgr = thumb

                        img[thumb_y:thumb_y+thumb_h, thumb_x:thumb_x+thumb_w] = thumb_bgr

                        cv2.rectangle(img, (thumb_x, thumb_y), (thumb_x + thumb_w - 1, thumb_y + thumb_h - 1), (100, 100, 255), 2)
                    except Exception:
                        pass

            cv2.putText(img, f"Cam{i}", (end_px, end_py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
    if tracked_objects:
        for i, obj in enumerate(tracked_objects):
            x = obj['x']
            y = obj['y']
            r = obj['range']
            
            if y >= 0:
                theta = math.atan2(y, x)
                
                r_plot = min(r, R_MAX)
                radius_px = int(round((r_plot / R_MAX) * R_px))
                
                px = cx + int(round(radius_px * math.cos(theta)))
                py = cy - int(round(radius_px * math.sin(theta)))
                
                px = max(0, min(debug_frame_w - 1, px))
                py = max(0, min(debug_frame_h - 1, py))
                
                cv2.circle(img, (px, py), 10, (0, 255, 0), 2)
                cv2.putText(img, "Kalman", (px - 30, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if obj['contour'] is not None:
                    contour_obj = obj['contour']
                    center_x = contour_obj['center_of_mass_x']
                    azimuth_camera = pixel_to_azimuth(center_x, camera_width_px, camera_fov_rad)
                    
                    theta_cam = math.pi / 2 - azimuth_camera
                    r_plot_cam = 0.5 * R_MAX
                    radius_px_cam = int(round((r_plot_cam / R_MAX) * R_px))
                    
                    px_cam = cx + int(round(radius_px_cam * math.cos(theta_cam)))
                    py_cam = cy - int(round(radius_px_cam * math.sin(theta_cam)))
                    
                    px_cam = max(0, min(debug_frame_w - 1, px_cam))
                    py_cam = max(0, min(debug_frame_h - 1, py_cam))
                    
                    cv2.line(img, (px, py), (px_cam, py_cam), (100, 100, 255), 1)

    legend_y = 30
    cv2.putText(img, "Radar (cyan)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    legend_y += 20
    cv2.putText(img, "Camera (magenta)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    legend_y += 20
    cv2.putText(img, "Kalman (green)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    legend_y += 20
    cv2.putText(img, "Association (orange)", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
    
    cv2.imshow(win, img)

