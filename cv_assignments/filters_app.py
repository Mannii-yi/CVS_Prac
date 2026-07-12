"""
Real-Time Instagram/Snapchat Style Filters
-------------------------------------------
Run: python filters_app.py

Keys:
  1 - Original
  2 - Sepia
  3 - Vintage (sepia + vignette + grain)
  4 - Sketch (pencil sketch)
  5 - Cartoon
  6 - Black & White
  7 - Warm tone
  8 - Cool tone
  9 - Thermal (CUSTOM filter)
  s - Save current frame to captures/
  q - Quit
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Filter functions — each takes a BGR frame and returns a BGR frame
# ---------------------------------------------------------------------------

def apply_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)


def apply_vignette(frame, strength=2.0):
    rows, cols = frame.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols / strength / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / strength / 2)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    vignette = np.copy(frame).astype(np.float32)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return np.clip(vignette, 0, 255).astype(np.uint8)


def apply_vintage(frame):
    sepia = apply_sepia(frame)
    vignette = apply_vignette(sepia, strength=1.6)
    # film grain
    noise = np.random.randint(0, 30, vignette.shape, dtype=np.uint8)
    grain = cv2.subtract(vignette, noise // 2)
    grain = cv2.add(grain, noise // 3)
    return grain


def apply_sketch(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_cartoon(frame):
    # Smooth colors but keep edges
    color = frame
    for _ in range(2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)
    return cartoon


def apply_bw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_warm(frame):
    b, g, r = cv2.split(frame.astype(np.int16))
    r = np.clip(r + 25, 0, 255)
    b = np.clip(b - 20, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def apply_cool(frame):
    b, g, r = cv2.split(frame.astype(np.int16))
    b = np.clip(b + 25, 0, 255)
    r = np.clip(r - 20, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def apply_thermal(frame):
    """CUSTOM / bonus filter — fake thermal camera look."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal


FILTERS = {
    ord('1'): ("Original", lambda f: f),
    ord('2'): ("Sepia", apply_sepia),
    ord('3'): ("Vintage", apply_vintage),
    ord('4'): ("Sketch", apply_sketch),
    ord('5'): ("Cartoon", apply_cartoon),
    ord('6'): ("Black & White", apply_bw),
    ord('7'): ("Warm Tone", apply_warm),
    ord('8'): ("Cool Tone", apply_cool),
    ord('9'): ("Thermal (Custom)", apply_thermal),
}


def draw_hud(frame, filter_name, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, f"Filter: {filter_name}", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "1-9 filters | s save | q quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    current_key = ord('1')
    prev_time = time.time()

    print("Running. Focus the video window and press keys 1-9 / s / q.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural selfie view

        name, func = FILTERS[current_key]
        try:
            output = func(frame)
        except Exception as e:
            print("Filter error:", e)
            output = frame

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        output = draw_hud(output, name, fps)
        cv2.imshow("CV Filters", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = os.path.join(
                SAVE_DIR, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            cv2.imwrite(fname, output)
            print(f"Saved {fname}")
        elif key in FILTERS:
            current_key = key

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()