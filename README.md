# Public Behavioral Anomaly Detection System

This repository contains the source code for a behavioral anomaly detection system designed for public spaces, . The system analyzes three key services: facial emotion, drowsiness, and physical posture to determine if a person is behaving abnormally in public. 

Persons are assigned unique Face IDs and tracked. The system captures at a smooth **30 FPS**, while performing deep behavioral analysis at **0.2 FPS** (every 5 seconds) to prevent ML service bottlenecking. Individuals exhibiting abnormal behavior for more than 5 minutes are flagged. If the abnormality persists for 10 minutes, their photo and relevant information are stored.

## Performance & Buffer System

To maintain a perfect balance between real-time behavioral monitoring and system performance without overwhelming ML services, we utilize a multi-tiered buffer architecture:

- **Capture & Display:** The system captures and displays video at **30 FPS** (`CAP_PROP_BUFFERSIZE=1` to prevent lag).
- **Deep Analysis Trigger:** Deep behavioral analysis runs at **0.2 FPS** (every 5 seconds, configurable between 3-10s).
- **Buffer Mechanisms employed:**
  - **Frame Buffer:** 1 frame size to prevent camera lag.
  - **Skip Buffer:** Drops stale frames (2 frames) before analysis to ensure the ML model always receives the most current data.
  - **History Buffer:** Maintains the last 10 analysis results.
  - **Temporal Buffer:** Stores 1200 frames (20 minutes at ~1 FPS) for backend temporal tracking of individuals.

*Net Result: Smooth real-time video feed combined with deep inference every 5 seconds, allowing stable API calls (~2-5s processing) alongside immediate UI updates.*

## Dataset Documentation

This project utilizes both public and synthetic datasets to train the behavioral models.

### Public Datasets (Facial Emotion & Drowsiness)

1. **Facial Emotion Dataset (FER-13)**
   - **Source:** https://www.kaggle.com/datasets/msambare/fer2013 
   - **Why it fits Behavioral Analytics:** Emotions provide critical non-verbal cues for assessing an individual's psychological state and intent in public, fitting perfectly into the behavioral analysis framework.
   - **Engineered Behavioral Features:** Raw grayscale images (48x48) are rescaled and processed by a CNN. The model focuses on the 7 basic emotions (Happy, Surprise, Angry, Disgust, Fear, Sad, Neutral).

2. **Drowsiness Dataset**
   - **Source:** Roboflow Driver Drowsiness Dataset - Using roboflow api-key
   - **Why it fits Behavioral Analytics:** Drowsiness and fatigue are behavioral indicators that can correlate with abnormalities such as loitering, distress, or impaired decision-making in public areas.
   - **Engineered Behavioral Features:** The YOLOv8 model extracts spatial features directly from bounding box annotations for the eyes and face to classify individuals into `awake` and `drowsy` states.

### Synthetic Dataset (Posture Analysis)

1. **Posture Dataset**
   - **Dataset Type:** Synthetic
   - **Why Synthetic Data:** There was no readily available real behavioral dataset that mapped comprehensive full-body anomalies sequentially for public monitoring.
   - **How it was generated:** The data was self-generated using Google's MediaPipe Pose. Specific behavioral dynamics were recorded by evaluating landmarks.
   - **Number of Records:** 
     - Normal behavior: 4,000 frames (e.g., normal walking, normal standing)
     - Anomaly behavior: 6,000 frames (e.g., stumbling, erratic movement, sudden direction change)
   - **Feature Description:** The dataset consists of 99 dimensions (33 landmarks x 3 coordinates) scaled down to 6 key geometric angles: Shoulder Pitch, Hip Yaw, Knee Flexion, Elbow Bend, Head Roll, and Spine Curvature. These represent behavioral dynamics over 60-frame windows (2 seconds at 30fps), normalized using Z-score standardization.

## Project Structure
The repository contains all core backend services, the primary execution script, and the pre-trained model weights. 

- **Service Files:** Contains all 5 source code files orchestrating the UI, video capture, emotion, drowsiness, and posture analysis services.
- **Model Weights & Scalers:**
  - `10-06-best_mini_xception_paper.h5`: Trained model weights for Emotion Detection.
  - `best.pt`: Optimized PyTorch weights for Drowsiness Detection (YOLOv8).
  - `behavioral_anomaly_model_final.h5`: Trained LSTM model for Posture tracking.
  - `pose_scaler.pkl`: Feature scaler used for Pose Normalization.
- **`run_all_services.py`**: The primary executable script to seamlessly initiate the tracker and all 3 integrated anomaly detection services.
- **`requirements.txt`**: Complete list of environment dependencies to replicate the project locally.

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the main anomaly detection interface
python run_all_services.py
```
