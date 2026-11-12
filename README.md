#  Hand Gesture Fruit Ninja (Python)

A real-time webcam hand-tracking arcade game built with **Python**, **Pygame**, **OpenCV**, and **Google MediaPipe Hands** (fully compatible with **Python 3.13+** MediaPipe Tasks API). Use your index fingertip in front of your webcam as a glowing blade to slice flying fruit while avoiding bombs!

##  Features
- **Real-Time Hand Tracking:** Tracks your index fingertip (MediaPipe landmark `8`) on a dedicated background thread for lag-free 60 FPS gameplay. Automatically downloads `hand_landmarker.task` on first run for Python 3.13+ compatibility.
- **Picture-in-Picture Camera Preview:** Live mirrored webcam feed embedded directly inside the game stage.
- **Arcade HUD & Controls:** Tracks `Score`, `60s Timer`, `3 Lives`, and persistent `Best Score`, with interactive `Enable camera`, `Pause/Resume` (or `Space`), and `Restart` controls.
- **Mouse Fallback:** Automatically supports mouse cursor slicing if a webcam is unavailable.

##  How to Run

### Option 1: One-Click Run (Windows)
Double-click **`run_game.bat`** inside this folder. It will automatically install the required libraries and launch the game.

### Option 2: Run via Terminal
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the game:
   ```bash
   python main.py
   ```
