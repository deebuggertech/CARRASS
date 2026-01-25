import os
import time

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any
import cv2
from typing import Optional

model = YOLO('../models/yolo.pt')

def classify_objects(
        objects: List[Dict[str, Any]],
        frame: np.ndarray,
        background_frame: Optional[np.ndarray] = None,
        verbose: bool = False
) -> List[Dict[str, Any]]:

    if frame is None or len(objects) == 0:
        return objects

    vis_frame = frame.copy() if verbose else None

    for obj in objects:
        obj['classification'] = 'unknown'

        if 'contour' in obj and obj['contour'] is not None:
            cnt = obj['contour'].get('contour')

            ox, oy, ow, oh = cv2.boundingRect(cnt)
            obj_crop = frame[oy:oy+oh, ox:ox+ow]

            if obj_crop.size > 0:
                results = model.predict(source=obj_crop, conf=0.4, verbose=False)

                best_conf = 0
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf > best_conf:
                            best_conf = conf
                            obj['classification'] = 'robot'

            if verbose:
                color = (0, 255, 0) if obj['classification'] == 'robot' else (0, 0, 255)
                cv2.rectangle(vis_frame, (ox, oy), (ox + ow, oy + oh), (255, 255, 0), 1)
                label = f"{obj['classification']} ({best_conf:.2f})" if best_conf > 0 else "unknown"
                cv2.putText(vis_frame, label, (ox, oy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if verbose:
        cv2.imshow("PRE-Classification", vis_frame)
        cv2.waitKey(1)

    return objects


def save_classified_crops(
        objects: List[Dict[str, Any]],
        frame: np.ndarray,
        background: np.ndarray,
        output_dir: str = "detections",
        padding: int = 20
):
    if frame is None or background is None or not objects:
        return

    session_id = int(time.time())
    session_path = os.path.join(output_dir, f"session_{session_id}")

    h_frame, w_frame = frame.shape[:2]

    for i, obj in enumerate(objects):
        contour_data = obj.get('contour')
        if contour_data is None:
            continue

        cnt = contour_data.get('contour')
        if cnt is None:
            continue

        ox, oy, ow, oh = cv2.boundingRect(cnt)

        x1, y1 = max(0, ox - padding), max(0, oy - padding)
        x2, y2 = min(w_frame, ox + ow + padding), min(h_frame, oy + oh + padding)

        crop_frame = frame[y1:y2, x1:x2]
        crop_bg = background[y1:y2, x1:x2]

        if crop_frame.size == 0:
            continue

        label = obj.get('classification', 'unknown')
        class_path = os.path.join(session_path, label)
        os.makedirs(class_path, exist_ok=True)

        base_name = f"obj_{i}_{session_id}"
        cv2.imwrite(os.path.join(class_path, f"{base_name}_target.jpg"), crop_frame)
        cv2.imwrite(os.path.join(class_path, f"{base_name}_bg.jpg"), crop_bg)
