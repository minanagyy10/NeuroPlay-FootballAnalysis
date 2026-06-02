# ⚽ Football AI Analysis System

> An end-to-end AI-powered football match analysis system that automatically detects players, tracks movement, estimates speed, detects events, separates teams, and generates tactical recommendations — all from a raw match video.

---

## 🎬 Demo Output

| Feature | Output |
|---|---|
| Player Detection | Bounding boxes on every player, goalkeeper, and referee |
| Player Tracking | Stable IDs maintained across the full match |
| Team Separation | Blue boxes for Team A, Green boxes for Team B |
| Speed Estimation | Real-time km/h displayed per player |
| Ball Detection | Ball tracked with trail overlay |
| Events | Pass, Shot, Cross, Possession Change detected |
| Heatmaps | Per-player and team position heatmaps |
| Tactical AI | xG, xT analysis with AI coach suggestions |
| Mini Radar | Top-down pitch view in corner of video |
| Possession Bar | Live possession percentage overlay |

---

## 🧠 Technologies Used

### Computer Vision & AI
| Technology | Purpose |
|---|---|
| **YOLOv8** (Ultralytics) | Player, ball, goalkeeper, referee detection |
| **ByteTrack** (Supervision) | Multi-object tracking across frames |
| **OpenCV** | Frame processing, homography, annotation |
| **K-Means Clustering** (Scikit-learn) | Automatic team color separation |
| **Histogram Comparison** | Player Re-Identification (ReID) |

### Deep Learning
| Technology | Purpose |
|---|---|
| **PyTorch** | Deep learning backend |
| **CUDA** | GPU acceleration (NVIDIA GPU) |
| **Transfer Learning** | Fine-tuned YOLOv8 on football dataset |
| **Roboflow** | Football dataset download and management |

### Analytics & Models
| Technology | Purpose |
|---|---|
| **xG Model** | Expected Goals — probability a shot results in a goal |
| **xT Model** | Expected Threat — value of ball position on pitch |
| **Homography** | Pixel-to-metres coordinate conversion |
| **Gaussian Blur** | Smooth heatmap generation |

### Backend & API
| Technology | Purpose |
|---|---|
| **FastAPI** | REST API for video upload and result retrieval |
| **Uvicorn** | ASGI server |
| **Python Multipart** | Video file upload handling |

### LLM Integration
| Technology | Purpose |
|---|---|
| **Claude API** (Anthropic) | Natural language tactical coaching advice |
| **HTTPX** | Async HTTP client for API calls |

### Frontend
| Technology | Purpose |
|---|---|
| **React** | Dashboard UI components |
| **Tailwind CSS** | Styling |
| **Recharts** | Speed and stats charts |

---

## 📁 Project Structure

```
football_ai/
│
├── run.py                        # Start here — configure and run pipeline
├── pipeline.py                   # Main orchestrator — wires all modules
├── train.py                      # Fine-tune YOLOv8 on football dataset
├── requirements.txt              # All dependencies
│
├── detection/
│   └── detector.py               # YOLOv8 player + ball detection
│
├── tracking/
│   └── tracker.py                # ByteTrack multi-object tracking
│
├── calibration/
│   └── homography.py             # Camera calibration — pixels to metres
│
├── analytics/
│   ├── speed.py                  # Speed estimation in km/h
│   ├── heatmap.py                # Position heatmaps + formation detection
│   └── events.py                 # Pass, shot, cross, event detection
│
├── tactical_ai/
│   └── coach.py                  # xG model, xT model, AI coach (Claude API)
│
├── utils/
│   ├── annotator.py              # Video frame annotation + team colors
│   ├── camera.py                 # Auto camera type detection
│   ├── reid.py                   # Player Re-Identification system
│   └── team_separator.py         # K-Means team color separation
│
├── backend/
│   └── main.py                   # FastAPI REST API server
│
└── frontend/
    └── Dashboard.jsx             # React analysis dashboard
```

---

## ⚙️ System Architecture

```
Upload Video
      ↓
Camera Auto-Detection (Broadcast / Drone / Fixed Wide / Fixed Close)
      ↓
Frame Extraction (OpenCV)
      ↓
Player + Ball Detection (YOLOv8 fine-tuned on football data)
      ↓
Multi-Object Tracking (ByteTrack)
      ↓
Player Re-Identification (Color Histogram Matching)
      ↓
Team Separation (K-Means on jersey colors)
      ↓
Speed + Distance Analysis (Homography + rolling median filter)
      ↓
Event Detection (Pass / Shot / Cross / Possession)
      ↓
Heatmap Generation + Formation Detection
      ↓
Tactical AI (xG + xT + Claude LLM coaching)
      ↓
Annotated Video + JSON Report + Heatmap Images
```

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.10+
- NVIDIA GPU (recommended) or CPU
- Git

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/football-ai.git
cd football-ai
```

### 3. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Install PyTorch (GPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:
```bash
pip install torch torchvision
```

### 5. Install all dependencies

```bash
pip install -r requirements.txt
```

### 6. Download a football video

Place any football match video in the project folder and rename it `match.mp4`.

For best results use:
- Broadcast TV side-view
- Tactical/drone camera
- Minimum 720p resolution
- Minimum 1 minute duration

### 7. Run the analysis

```bash
python run.py
```

### 8. View results

```
output/
├── annotated.mp4        ← video with all overlays
├── report.json          ← full stats in JSON
└── heatmaps/
    ├── team_heatmap.jpg
    ├── player_1_heatmap.jpg
    ├── player_2_heatmap.jpg
    └── ...
```

---

## 🔧 Configuration

Open `run.py` and edit these settings:

```python
# Your video file
VIDEO_PATH = "match.mp4"

# Your trained model (after running train.py)
model_path = "runs/detect/runs/train/football_ball-3/weights/best.pt"

# Device
device = "cuda"    # GPU (recommended)
device = "cpu"     # CPU (slower)

# Camera calibration (optional — for accurate km/h)
pixel_pts = None   # set field corner pixel coordinates
world_pts = None   # set field corner real-world metres

# Optional: Claude API for AI coaching
anthropic_api_key = None   # paste your key here
```

---

## 🎯 Training Your Own Model

The base YOLOv8 model is trained on COCO data. For better football detection (especially the ball), fine-tune on football-specific data:

### 1. Get a Roboflow API key

Sign up free at [roboflow.com](https://roboflow.com) and copy your API key.

### 2. Add your key to train.py

```python
rf = Roboflow(api_key="YOUR_KEY_HERE")
```

### 3. Run training

```bash
python train.py
```

Training takes 20–40 minutes on an NVIDIA GPU. The best model is saved to:
```
runs/detect/runs/train/football_ball-3/weights/best.pt
```

### 4. Update run.py

```python
model_path = r"runs\detect\runs\train\football_ball-3\weights\best.pt"
```

---

## 🌐 API Server

Start the REST API server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the interactive docs at:
```
http://localhost:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check server status |
| `POST` | `/analyse` | Upload video and start analysis |
| `GET` | `/status/{job_id}` | Poll analysis progress |
| `GET` | `/report/{job_id}` | Get full JSON report |
| `GET` | `/video/{job_id}` | Download annotated video |
| `GET` | `/heatmap/{job_id}/{player_id}` | Get player heatmap |

### Example Usage

```python
import requests

# Upload video
with open("match.mp4", "rb") as f:
    response = requests.post("http://localhost:8000/analyse",
        files={"video": f})

job_id = response.json()["job_id"]

# Poll status
status = requests.get(f"http://localhost:8000/status/{job_id}").json()

# Get report when complete
report = requests.get(f"http://localhost:8000/report/{job_id}").json()
print(report["possession"])
print(report["speed_stats"])
```

---

## 📊 Output Report Structure

```json
{
  "video": "match.mp4",
  "duration_s": 5760.0,
  "unique_players": 28,
  "possession": {
    "Team A": 54.3,
    "Team B": 45.7
  },
  "team_a_players": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
  "team_b_players": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
  "formation": "4-3-3",
  "events": {
    "PASS": 487,
    "SHOT": 24,
    "CROSS": 31,
    "POSSESSION_CHANGE": 203
  },
  "speed_stats": [
    {
      "player_id": 7,
      "max_speed_kmh": 33.2,
      "avg_speed_kmh": 8.1,
      "distance_m": 9823,
      "sprint_time_s": 42
    }
  ],
  "tactical_log": [
    {
      "t": 312.5,
      "action": "SHOOT",
      "xg": 0.31,
      "xt": 0.441,
      "reason": "High-value shooting position inside penalty box."
    }
  ]
}
```

---

## 🤖 AI Tactical Coach

Enable the Claude-powered tactical coach by adding your Anthropic API key:

```python
anthropic_api_key = "your-anthropic-key"
```

Get a free key at [console.anthropic.com](https://console.anthropic.com).

The coach analyses every 25 frames and generates advice like:

```
"Player has the ball at (78m, 34m) with xG=0.31.
 Three defenders are closing but a teammate is free
 on the left with xT=0.44. Through pass is the
 optimal decision to create a high-probability chance."
```

---

## 📈 Performance

Tested on NVIDIA RTX 3050 Ti Laptop GPU:

| Video Resolution | Processing Speed |
|---|---|
| 576×576 | ~37 fps |
| 640×360 | ~37 fps |
| 1280×720 | ~12 fps |
| 1920×1080 | ~6 fps |

For faster processing on long matches, set `frame_skip = 2` in `run.py` to process every other frame.

---

## 🗺️ Roadmap

- [x] Player detection (YOLOv8)
- [x] Multi-object tracking (ByteTrack)
- [x] Player Re-Identification (ReID)
- [x] Team color separation (K-Means)
- [x] Speed estimation
- [x] Heatmap generation
- [x] Event detection (pass, shot, cross)
- [x] Formation detection
- [x] Tactical AI (xG, xT, Claude)
- [x] REST API (FastAPI)
- [x] Camera auto-detection
- [ ] Jersey number recognition (OCR)
- [ ] Offside detection
- [ ] Automatic highlights generation
- [ ] Real-time processing
- [ ] Team formation comparison
- [ ] Player comparison dashboard

---

## 📦 Dependencies

```
torch >= 2.1.0
torchvision >= 0.16.0
ultralytics >= 8.2.0
supervision >= 0.21.0
opencv-python >= 4.9.0
fastapi >= 0.110.0
uvicorn >= 0.29.0
scikit-learn >= 1.3.0
httpx >= 0.27.0
numpy >= 1.26.0
matplotlib >= 3.8.0
scipy >= 1.13.0
roboflow >= 1.3.0
python-dotenv >= 1.0.0
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [Supervision](https://github.com/roboflow/supervision) — tracking utilities
- [Roboflow](https://roboflow.com) — football dataset
- [SoccerNet](https://www.soccer-net.org) — football analytics research
- [StatsBomb](https://statsbomb.com) — xG/xT methodology inspiration
- [Anthropic Claude](https://anthropic.com) — tactical AI coaching

---

## 👨‍💻 Author

Built as a graduation project demonstrating the application of computer vision, deep learning, and AI to sports analytics.

> *"The best teams in the world use data. Now you can too."*
