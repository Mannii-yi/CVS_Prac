"""
AI Proctoring System
---------------------
Run: python proctoring_system.py

Requires: opencv-python, mediapipe   (pip install opencv-python mediapipe)
Optional bonus: ultralytics (pip install ultralytics) for phone detection.
If ultralytics / the yolo weights aren't available, the script still runs
fine with phone detection simply disabled.

Detects:
  - No face visible
  - Multiple faces
  - Looking away for too long (head pose via solvePnP)
  - (Bonus) Phone / suspicious object in frame via YOLOv8

Produces:
  - Live on-screen alerts
  - A timestamped event log written to proctoring_log.csv
  - A short summary report printed at the end
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime
from collections import Counter

# --- try to enable the YOLO bonus, degrade gracefully if unavailable ---
YOLO_ENABLED = True
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")  # auto-downloads on first run if internet available
except Exception as e:
    print(f"[info] YOLO bonus disabled ({e}). Continuing without phone detection.")
    YOLO_ENABLED = False

# ---------------------------------------------------------------------------
# Config / thresholds — tune these to taste
# ---------------------------------------------------------------------------
LOOK_AWAY_YAW_THRESHOLD = 25      # degrees
LOOK_AWAY_PITCH_THRESHOLD = 20    # degrees
LOOK_AWAY_TIME_LIMIT = 3.0        # seconds before it counts as an alert
NO_FACE_TIME_LIMIT = 2.0          # seconds
YOLO_EVERY_N_FRAMES = 15          # run YOLO occasionally, it's expensive
LOG_FILE = "proctoring_log.csv"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 3D model points of a generic face (used for solvePnP head pose)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# corresponding mediapipe FaceMesh landmark indices
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class EventLogger:
    def __init__(self, path):
        self.path = path
        self.rows = []
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "event", "details"])

    def log(self, event, details=""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, event, details])
        self.rows.append((ts, event, details))
        print(f"[{ts}] {event} {details}")

    def summary(self):
        counts = Counter(r[1] for r in self.rows)
        return counts


def get_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]
    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in LANDMARK_IDS],
        dtype=np.float64,
    )

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj_matrix = np.hstack((rotation_mat, np.zeros((3, 1))))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [float(a) for a in euler_angles]
    return yaw, pitch


def detect_phone(frame):
    if not YOLO_ENABLED:
        return False
    results = yolo_model.predict(frame, verbose=False, conf=0.4)
    for r in results:
        for box in r.boxes:
            cls_name = yolo_model.names[int(box.cls[0])]
            if cls_name == "cell phone":
                return True
    return False


def draw_alert_banner(frame, text, color=(0, 0, 255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 45), (w, h), color, -1)
    cv2.putText(frame, text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    logger = EventLogger(LOG_FILE)
    logger.log("SESSION_START")

    no_face_start = None
    look_away_start = None
    frame_count = 0
    phone_flag = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        alert_text = ""
        alert_color = (0, 150, 0)

        faces = results.multi_face_landmarks or []
        num_faces = len(faces)

        # --- no face / multiple faces ---
        if num_faces == 0:
            if no_face_start is None:
                no_face_start = time.time()
            elif time.time() - no_face_start > NO_FACE_TIME_LIMIT:
                alert_text = "ALERT: No face detected"
                alert_color = (0, 0, 255)
                logger.log("NO_FACE")
                no_face_start = time.time()  # reset so we don't spam log every frame
        else:
            no_face_start = None

        if num_faces > 1:
            alert_text = f"ALERT: {num_faces} faces detected"
            alert_color = (0, 0, 255)
            logger.log("MULTIPLE_FACES", details=f"count={num_faces}")

        # --- looking away (only meaningful with exactly one face) ---
        if num_faces == 1:
            yaw, pitch = get_head_pose(faces[0].landmark, frame.shape)
            if yaw is not None:
                looking_away = (
                    abs(yaw) > LOOK_AWAY_YAW_THRESHOLD
                    or abs(pitch) > LOOK_AWAY_PITCH_THRESHOLD
                )
                if looking_away:
                    if look_away_start is None:
                        look_away_start = time.time()
                    elif time.time() - look_away_start > LOOK_AWAY_TIME_LIMIT:
                        alert_text = "ALERT: Looking away too long"
                        alert_color = (0, 0, 255)
                        logger.log("LOOKING_AWAY", details=f"yaw={yaw:.1f} pitch={pitch:.1f}")
                        look_away_start = time.time()
                else:
                    look_away_start = None

            # draw landmarks lightly for demo value
            for i in LANDMARK_IDS:
                lm = faces[0].landmark[i]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 255), -1)

        # --- bonus: phone detection every N frames ---
        if YOLO_ENABLED and frame_count % YOLO_EVERY_N_FRAMES == 0:
            phone_flag = detect_phone(frame)
            if phone_flag:
                logger.log("PHONE_DETECTED")

        if phone_flag:
            alert_text = "ALERT: Phone detected"
            alert_color = (0, 0, 255)

        if alert_text:
            draw_alert_banner(frame, alert_text, alert_color)
        else:
            draw_alert_banner(frame, "Status: OK", (0, 130, 0))

        cv2.putText(frame, f"Faces: {num_faces}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("AI Proctoring System", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logger.log("SESSION_END")
    cap.release()
    cv2.destroyAllWindows()

    # --- final report ---
    counts = logger.summary()
    print("\n===== SESSION REPORT =====")
    for event, n in counts.items():
        print(f"{event}: {n}")
    print(f"Full log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()