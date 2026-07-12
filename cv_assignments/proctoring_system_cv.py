"""
AI Proctoring System (OpenCV-only version — no mediapipe / matplotlib needed)
------------------------------------------------------------------------------
Run: python proctoring_system_cv.py

Only needs: opencv-python  (already installed)
Optional bonus: ultralytics, for phone detection.

Detects:
  - No face visible
  - Multiple faces
  - Looking away (frontal face lost but a side/profile face is detected)
  - (Bonus) Phone detection via YOLOv8, if ultralytics is installed

Produces:
  - Live on-screen alerts
  - proctoring_log.csv event log
  - End-of-session summary report
"""

import cv2
import time
import csv
from datetime import datetime
from collections import Counter

YOLO_ENABLED = True
try:
    from ultralytics import YOLO
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"[info] YOLO bonus disabled ({e}). Continuing without phone detection.")
    YOLO_ENABLED = False

# ---------------------------------------------------------------------------
NO_FACE_TIME_LIMIT = 2.0        # seconds before "no face" alert
LOOK_AWAY_TIME_LIMIT = 3.0      # seconds before "looking away" alert
YOLO_EVERY_N_FRAMES = 15
LOG_FILE = "proctoring_log.csv"

# Haar cascades ship inside opencv-python — no extra downloads needed
frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)


class EventLogger:
    def __init__(self, path):
        self.path = path
        self.rows = []
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "event", "details"])

    def log(self, event, details=""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([ts, event, details])
        self.rows.append((ts, event, details))
        print(f"[{ts}] {event} {details}")

    def summary(self):
        return Counter(r[1] for r in self.rows)


def detect_phone(frame):
    if not YOLO_ENABLED:
        return False
    results = yolo_model.predict(frame, verbose=False, conf=0.4)
    for r in results:
        for box in r.boxes:
            if yolo_model.names[int(box.cls[0])] == "cell phone":
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

    print("Running. Press q in the video window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        frontal_faces = frontal_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
        )
        profile_faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
        )

        num_frontal = len(frontal_faces)
        alert_text = ""
        alert_color = (0, 130, 0)

        # --- no face at all (neither frontal nor profile) ---
        if num_frontal == 0 and len(profile_faces) == 0:
            if no_face_start is None:
                no_face_start = time.time()
            elif time.time() - no_face_start > NO_FACE_TIME_LIMIT:
                alert_text = "ALERT: No face detected"
                alert_color = (0, 0, 255)
                logger.log("NO_FACE")
                no_face_start = time.time()
        else:
            no_face_start = None

        # --- multiple faces ---
        if num_frontal > 1:
            alert_text = f"ALERT: {num_frontal} faces detected"
            alert_color = (0, 0, 255)
            logger.log("MULTIPLE_FACES", details=f"count={num_frontal}")

        # --- looking away: frontal face lost but a profile face is present ---
        if num_frontal == 0 and len(profile_faces) > 0:
            if look_away_start is None:
                look_away_start = time.time()
            elif time.time() - look_away_start > LOOK_AWAY_TIME_LIMIT:
                alert_text = "ALERT: Looking away too long"
                alert_color = (0, 0, 255)
                logger.log("LOOKING_AWAY")
                look_away_start = time.time()
        else:
            look_away_start = None

        for (x, y, w, h) in frontal_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        for (x, y, w, h) in profile_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 150, 0), 2)

        # --- bonus phone detection ---
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

        cv2.putText(frame, f"Faces: {num_frontal}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("AI Proctoring System", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    logger.log("SESSION_END")
    cap.release()
    cv2.destroyAllWindows()

    counts = logger.summary()
    print("\n===== SESSION REPORT =====")
    for event, n in counts.items():
        print(f"{event}: {n}")
    print(f"Full log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
