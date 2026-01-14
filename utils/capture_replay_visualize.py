import argparse

from data_sources.camera_capture import CameraCapture
from data_sources.radar_capture import RadarCapture
from data_sources.storage import DataRecorder, DataPlayer
import time
import cv2
import numpy as np
import math

RADAR_WIN = 'Radar'
CAMERA_WIN = 'Camera Frame'
RECORDINGS_DIR = 'recordings'
W, H = 1000, 500
R_MAX = 3.0


def visualize_camera(frame, win=CAMERA_WIN):
    if frame is None:
        placeholder = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'No camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        cv2.imshow(win, placeholder)
    else:
        h, w = frame.shape[:2]
        s = min(W/w, H/h)
        nw, nh = int(w*s), int(h*s)
        resized = cv2.resize(frame, (nw, nh))
        top, left = (H - nh) // 2, (W - nw) // 2
        frame = cv2.copyMakeBorder(resized, top, H-nh-top, left, W-nw-left, cv2.BORDER_CONSTANT)
        cv2.imshow(win, frame)


def visualize_radar(radar_data, win=RADAR_WIN):
    img = np.zeros((H, W, 3), dtype=np.uint8)

    margin = 20
    R_px = min(W, H) // 2 - margin
    cx, cy = W // 2, H // 2
    cv2.ellipse(img, (cx, cy), (R_px, R_px), 0, 180, 360, (30, 30, 30), 1)

    for user_angle in (+50.0, -50.0):
        std_deg = user_angle + 90.0
        theta_rad = math.radians(std_deg)
        end_x = cx + int(round(R_px * math.cos(theta_rad)))
        end_y = cy - int(round(R_px * math.sin(theta_rad)))
        cv2.line(img, (cx, cy), (end_x, end_y), (0, 200, 0), 2)
        cv2.circle(img, (end_x, end_y), 4, (0, 200, 0), -1)

    def map_to_pixel_polar(x, y):
        try:
            r = math.hypot(x, y)
            theta = math.atan2(y, x)
        except Exception:
            return None

        r_plot = min(r, R_MAX)
        radius_px = int(round((r_plot / R_MAX) * R_px))

        px = cx + int(round(radius_px * math.cos(theta)))
        py = cy - int(round(radius_px * math.sin(theta)))

        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))

        return px, py, r, math.degrees(theta)

    if radar_data:
        try:
            if isinstance(radar_data, dict) and 'x' in radar_data and 'y' in radar_data:
                x = float(radar_data['x'])
                y = float(radar_data['y'])
                if y >= 0:
                    mapped = map_to_pixel_polar(x, y)
                    if mapped:
                        px, py, r, theta_deg = mapped
                        cv2.circle(img, (px, py), 6, (0, 255, 255), -1)
                        theta_user = theta_deg - 90.0
                        if theta_user > 180.0:
                            theta_user -= 360.0
                        if theta_user <= -180.0:
                            theta_user += 360.0
                        coord_text = f"r={r:.2f} m, angle={theta_user:+.1f} deg."
                        cv2.putText(img, coord_text, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        except Exception:
            pass

    cv2.imshow(win, img)


def capture_visualize(mode, recording_name):
    print("Initializing")

    if mode == 'record':
        recorder = DataRecorder(RECORDINGS_DIR, session_name=recording_name)
        player = None
    else:
        recorder = None
        player = DataPlayer(RECORDINGS_DIR+f'/{recording_name}')

    camera = CameraCapture(camera_index=0, recorder=recorder, player=player, capture_fps=4)
    radar = RadarCapture(port="COM12", recorder=recorder, player=player, capture_fps=4)

    print("Starting")

    camera.start()
    radar.start()

    cv2.namedWindow(RADAR_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(RADAR_WIN, W, H)
    cv2.namedWindow(CAMERA_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WIN, W, H)

    try:
        while True:
            frame = camera.get_frame()
            radar_data = radar.get_radar_data()

            visualize_camera(frame, CAMERA_WIN)
            visualize_radar(radar_data, RADAR_WIN)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if cv2.getWindowProperty(RADAR_WIN, cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty(CAMERA_WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            time.sleep(0.1)

    finally:
        try:
            camera.stop()
            radar.stop()
            cv2.destroyWindow(RADAR_WIN)
            cv2.destroyWindow(CAMERA_WIN)
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Visualization Tool")
    parser.add_argument('--mode', type=str, choices=['record', 'play'], default='record')
    parser.add_argument('--name', type=str, default=None)
    args = parser.parse_args()

    recording_name = args.name

    if args.mode == 'play':
        if recording_name is None:
            parser.error("--name required for play mode")

    capture_visualize(mode=args.mode, recording_name=args.name)