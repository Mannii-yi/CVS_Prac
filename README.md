# 👁️ Vision Lab: Filters & Focus

Two computer vision mini-apps built with OpenCV — one for fun, one for focus.

---

## 🎨 Project 1: Live Filters
Turn your webcam into a filter factory. Sepia, Vintage, Sketch, Cartoon, B&W, Warm/Cool tones — plus a custom **Thermal Cam** mode because why not.

```bash
python filters_app.py
```

| Key | Action |
|---|---|
| `1`–`9` | Switch filters |
| `s` | Save snapshot |
| `q` | Quit |

Snapshots land in `captures/`.

---

## 🕵️ Project 3: AI Proctor
A quiet watcher for online exams. Flags when you vanish, multiply, or wander off-screen — logs everything, judges no one (except maybe your phone).

```bash
python proctoring_system_cv.py
```

**Watches for:**
- 🚫 No face
- 👥 Multiple faces
- 👀 Looking away too long
- 📱 Phone in frame *(bonus, via YOLOv8)*

Every event gets timestamped in `proctoring_log.csv`, with a summary report printed when you press `q`.

---

## ⚙️ Under the Hood
Built entirely on **OpenCV** — no heavyweight ML frameworks required.

- **Filters:** color transforms, bilateral filtering, adaptive thresholding, colormaps
- **Proctoring:** dual Haar-cascade detection (frontal + profile) as a lightweight stand-in for landmark-based head-pose tracking, chosen for stability over precision

## 🚀 Setup
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Built in one very long night, fueled by stubbornness and one extremely frustrating dependency chase. 🎬
