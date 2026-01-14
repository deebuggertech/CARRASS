import argparse
import time
import cv2
import numpy as np
import math

from data_sources.camera_capture import CameraCapture
from data_sources.radar_capture import RadarCapture
from data_sources.storage import DataRecorder, DataPlayer
from processing.cam_background_removal import remove_background, load_background_image
from processing.classification import classify
from processing.kalman import KalmanFilter2D, track_objects, visualize_kalman
from utils.config import CAMERA_FOV_DEGREE

RADAR_WIN = 'Radar'
CAMERA_WIN = 'Camera Frame'
OUTPUT_WIN = 'Output'
RECORDINGS_DIR = 'recordings'
DEBUG_FRAME_W, DEBUG_FRAME_H = 800, 600

def visualize_camera(frame, win=CAMERA_WIN):
    if frame is None:
        placeholder = np.zeros((DEBUG_FRAME_H, DEBUG_FRAME_W, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'No camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        cv2.imshow(win, placeholder)
    else:
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        h, w = frame.shape[:2]
        s = min(DEBUG_FRAME_W / w, DEBUG_FRAME_H / h)
        nw, nh = int(w*s), int(h*s)
        resized = cv2.resize(frame, (nw, nh))
        top, left = (DEBUG_FRAME_H - nh) // 2, (DEBUG_FRAME_W - nw) // 2
        display_frame = cv2.copyMakeBorder(resized, top, DEBUG_FRAME_H - nh - top, left, DEBUG_FRAME_W - nw - left, cv2.BORDER_CONSTANT)
        cv2.imshow(win, display_frame)


def visualize_radar(radar_data, win=RADAR_WIN):
    img = np.zeros((DEBUG_FRAME_H, DEBUG_FRAME_W, 3), dtype=np.uint8)
    R_MAX = 4.0
    
    margin = 20
    R_px = min(DEBUG_FRAME_W, DEBUG_FRAME_H) // 2 - margin
    cx, cy = DEBUG_FRAME_W // 2, DEBUG_FRAME_H // 2
    cv2.ellipse(img, (cx, cy), (R_px, R_px), 0, 180, 360, (30, 30, 30), 1)
    
    for user_angle in (CAMERA_FOV_DEGREE / 2, -CAMERA_FOV_DEGREE / 2):
        std_deg = user_angle + 90.0
        theta_rad = math.radians(std_deg)
        end_x = cx + int(round(R_px * math.cos(theta_rad)))
        end_y = cy - int(round(R_px * math.sin(theta_rad)))
        cv2.line(img, (cx, cy), (end_x, end_y), (0, 200, 0), 2)
        cv2.circle(img, (end_x, end_y), 4, (0, 200, 0), -1)
    
    if radar_data:
        try:
            if isinstance(radar_data, dict) and 'x' in radar_data and 'y' in radar_data:
                x = float(radar_data['x'])
                y = float(radar_data['y'])
                if y >= 0:
                    r = math.hypot(x, y)
                    theta = math.atan2(y, x)
                    
                    r_plot = min(r, R_MAX)
                    radius_px = int(round((r_plot / R_MAX) * R_px))
                    
                    px = cx + int(round(radius_px * math.cos(theta)))
                    py = cy - int(round(radius_px * math.sin(theta)))
                    
                    px = max(0, min(DEBUG_FRAME_W - 1, px))
                    py = max(0, min(DEBUG_FRAME_H - 1, py))
                    
                    cv2.circle(img, (px, py), 6, (0, 255, 255), -1)
                    
                    theta_deg = math.degrees(theta)
                    theta_user = theta_deg - 90.0
                    if theta_user > 180.0:
                        theta_user -= 360.0
                    if theta_user <= -180.0:
                        theta_user += 360.0
                    coord_text = f"r={r:.2f} m, angle={theta_user:+.1f} deg."
                    cv2.putText(img, coord_text, (10, DEBUG_FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        except Exception:
            pass
    
    cv2.imshow(win, img)


def main_loop(mode, name, verbose):
    print("Initializing...")
    
    if mode == 'record':
        recorder = DataRecorder(RECORDINGS_DIR, session_name=name)
        player = None
    else:
        recorder = None
        player = DataPlayer(RECORDINGS_DIR + f'/{name}')
    
    camera = CameraCapture(camera_index=0, recorder=recorder, player=player, capture_fps=4)
    radar = RadarCapture(port="COM12", recorder=recorder, player=player, capture_fps=4)
    
    print("Starting...")
    camera.start()
    radar.start()

    cv2.namedWindow(RADAR_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(RADAR_WIN, DEBUG_FRAME_W, DEBUG_FRAME_H)
    cv2.namedWindow(CAMERA_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WIN, DEBUG_FRAME_W, DEBUG_FRAME_H)
    cv2.namedWindow(OUTPUT_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(OUTPUT_WIN, DEBUG_FRAME_W, DEBUG_FRAME_H)

    background = load_background_image("recordings/session_person/background_reference.jpg")
    
    trackers = []
    
    try:
        while True:
            camera_frame = camera.get_frame()
            radar_data = radar.get_radar_data()

            visualize_radar(radar_data, RADAR_WIN)
            visualize_camera(camera_frame, CAMERA_WIN)
            

            if camera_frame is not None:
                contour_data = remove_background(camera_frame, background, verbose=False)
                objects = track_objects(radar_data, contour_data, trackers, camera_width_px=camera_frame.shape[1], original_frame=camera_frame, verbose=verbose)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                try:
                    if (cv2.getWindowProperty(RADAR_WIN, cv2.WND_PROP_VISIBLE) < 1 or
                        cv2.getWindowProperty(CAMERA_WIN, cv2.WND_PROP_VISIBLE) < 1 or
                        cv2.getWindowProperty(OUTPUT_WIN, cv2.WND_PROP_VISIBLE) < 1):
                        break
                except Exception:
                    break
            
            time.sleep(0.25)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print("Shutting down...")
        try:
            camera.stop()
            radar.stop()
            if verbose:
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML-CARRASS")
    parser.add_argument('--mode', type=str, choices=['record', 'play'], default='record')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == 'play' and args.name is None:
        parser.error("--name required for play mode")

    main_loop(mode=args.mode, name=args.name, verbose=args.verbose)
