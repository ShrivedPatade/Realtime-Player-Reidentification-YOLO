# ⚽ Real-Time Player Re-Identification with YOLOv8 + OSNet

This project performs real-time multi-player tracking and re-identification using:
- **YOLOv8** for player detection
- **OSNet** from [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) for visual re-ID
- A **Hungarian + Centroid matching** algorithm for robust identity assignment
- **Multithreaded** video processing for performance optimization

---

## 📁 Folder Structure

```

.
├── deep-person-reid/            # Cloned repo (OSNet Re-ID)
├── modules/                     # Modular components (reader, detector, embedder, tracker)
├── proj/
│   ├── input/                   # Input videos
│   └── weights/                # YOLOv8 weights
│       └── best.pt
├── track.py                     # Main tracking script
├── setup\_env.bat               # Environment setup script (Windows)
└── requirements.txt

````

---

## 🚀 Setup Instructions

### 1. ✅ Prerequisites
- Python 3.8+
- Git
- Internet connection to download weights

### 2. ▶️ Run Setup

Open a terminal in the project root and run:

```bat
setup_env.bat
````

This will:

* Clone the `deep-person-reid` repository
* Create and activate a virtual environment
* Install all required dependencies
* Install `deep-person-reid` in editable mode
* Download the YOLOv8 model to `proj/weights/best.pt`

---

## 🎯 How It Works

1. **YOLOv8** detects players in each frame.
2. Each player crop is passed to **OSNet** to generate a ReID embedding.
3. **Hungarian + Centroid Matching** assigns consistent IDs using both:

   * Visual similarity (cosine distance of embeddings)
   * Spatial proximity (centroid distance)
4. Player ID history is stored and updated using a deque of embeddings.

---

## ▶️ Run the Tracker

After environment setup, activate it:

```bat
call env\Scripts\activate
```

Then run the main script:

```bash
python track.py
```

Press **Q** to quit the video window.

---

## 📦 Adding Your Own Model / Video

* **Replace Video**: Drop your video into `proj/input/` and update the path in `track.py`
* **Replace YOLOv8 Model**: Drop a new `best.pt` into `proj/weights/`

---

## 🛠 Requirements

If needed manually, install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🙋 Common Issues

* **Slow performance?** Use GPU (ensure CUDA is installed) and reduce resolution.
* **Model not loading?** Ensure you have a valid `.pt` file in `proj/weights/`
* **Python errors during setup?** Make sure `pip`, `setuptools`, and `wheel` are up to date.

---

## 📌 Credits

* **YOLOv8** by [Ultralytics](https://github.com/ultralytics/ultralytics)
* **OSNet / deep-person-reid** by [Kaiyang Zhou](https://github.com/KaiyangZhou/deep-person-reid)

```

---

Let me know if you'd like this version exported to a file (`README.md`) or styled for GitHub Pages.
```
