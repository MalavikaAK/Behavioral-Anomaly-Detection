from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import asyncio
import time
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import base64
from PIL import Image
import io
from collections import deque, defaultdict, Counter
from contextlib import asynccontextmanager

# Import facenet-pytorch for face embeddings
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

# Device for PyTorch (CPU or GPU, if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Globals initialized in lifespan
human_detection_model = None
facenet_model = None

# Temporal pattern analysis buffer keyed by face_id assigned by the tracker
#variables required for the tracker
temporal_buffers = defaultdict(lambda: {
    'anomaly_history': deque(maxlen=1200),  # 20 minutes at 1fps
    'total_frames': 0,
    'anomaly_count': 0,
    'last_alert_time': 0,
    'creation_time': time.time(),
    'dominant_anomalies': Counter(),
    'face_image_base64': None,
    'behavioral_timeline': deque(maxlen=100)
})

# Service URLs
EMOTION_SERVICE_URL = "http://localhost:8001"
DROWSINESS_SERVICE_URL = "http://localhost:8002"
POSE_SERVICE_URL = "http://localhost:8004"

# Pydantic Models
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    timestamp: Optional[float] = None

class TemporalAlert(BaseModel):
    face_id: int
    alert_type: str
    anomaly_pattern: str
    duration_minutes: float
    anomaly_percentage: float
    dominant_anomalies: List[str]
    face_image: Optional[str] = None
    confidence: float
    timestamp: float

class FaceAnalysisResult(BaseModel):
    face_id: int
    bbox: List[int]
    emotion_result: Optional[dict] = None
    drowsiness_result: Optional[dict] = None
    pose_result: Optional[dict] = None
    overall_status: str
    warning_level: str
    warning_color: str
    confidence: float
    analysis_summary: str
    anomaly_types: Optional[List[str]] = None
    temporal_alert: Optional[TemporalAlert] = None

class MultiFaceBehavioralResponse(BaseModel):
    total_faces: int
    faces_analysis: List[FaceAnalysisResult]
    overall_scene_status: str
    scene_warning_color: str
    timestamp: float
    temporal_alerts: List[TemporalAlert] = []

# FastAPI lifespan handler replacing deprecated startup event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global human_detection_model, facenet_model
    try:
        from ultralytics import YOLO
        human_detection_model = YOLO('yolov8n.pt')
        print("✅ Human detection model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load human detection model: {e}")
        human_detection_model = None
    try:
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print("✅ FaceNet-pytorch embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load FaceNet-pytorch model: {e}")
        facenet_model = None
    yield
    # Add shutdown cleanup here if necessary

app = FastAPI(title="Behavioral Analysis Service with FaceNet-PyTorch", version="2.0.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def detect_humans_first(image_data):
    global human_detection_model
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if human_detection_model is None:
            print("⚠️ Human detection model not available, proceeding with analysis")
            return True, opencv_image
        results = human_detection_model(opencv_image, classes=[0], conf=0.3, verbose=False)
        humans_detected = len(results[0].boxes) > 0 if results[0].boxes is not None else False
        if humans_detected:
            human_count = len(results[0].boxes)
            print(f" {human_count} human(s) detected - proceeding with analysis")
        else:
            print(" No humans detected - skipping processing")
        return humans_detected, opencv_image
    except Exception as e:
        print(f"Human detection error: {e}")
        return True, opencv_image

def get_face_embedding(face_rgb):
    global facenet_model, device
    if facenet_model is None:
        return None
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160,160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet_model(face_tensor)
        return embedding.cpu().numpy()[0]
    except Exception as e:
        print(f"Embedding extraction error: {e}")
        return None

def cosine_similarity(a, b):
    if a is None or b is None:
        return -1
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

class SimpleTracker:
    def __init__(self, max_disappeared=30, similarity_threshold=0.6):
        self.next_face_id = 1
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.similarity_threshold = similarity_threshold
        self.frame_idx = 0

    def update(self, detections):
        self.frame_idx += 1
        assigned_tracks = []
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())

        if len(self.tracks) == 0:
            for i, det in enumerate(detections):
                face_id = self.next_face_id
                self.next_face_id += 1
                self.tracks[face_id] = {'embedding': det['embedding'], 'bbox': det['bbox'], 'last_seen': self.frame_idx}
                assigned_tracks.append({'face_id': face_id, 'bbox': det['bbox'], 'embedding': det['embedding']})
            return assigned_tracks

        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))

        for i, tid in enumerate(track_ids):
            track_emb = self.tracks[tid]['embedding']
            for j, det in enumerate(detections):
                cost_matrix[i, j] = cosine_similarity(track_emb, det['embedding'])

        assigned_detections = set()
        for _ in range(min(len(track_ids), len(detections))):
            i, j = divmod(np.argmax(cost_matrix), cost_matrix.shape[1])
            max_sim = cost_matrix[i, j]
            if max_sim < self.similarity_threshold:
                break
            face_id = track_ids[i]
            det = detections[j]
            self.tracks[face_id] = {'embedding': det['embedding'], 'bbox': det['bbox'], 'last_seen': self.frame_idx}
            assigned_tracks.append({'face_id': face_id, 'bbox': det['bbox'], 'embedding': det['embedding']})
            assigned_detections.add(j)
            cost_matrix[i, :] = -1
            cost_matrix[:, j] = -1

        for idx, det in enumerate(detections):
            if idx not in assigned_detections:
                face_id = self.next_face_id
                self.next_face_id += 1
                self.tracks[face_id] = {'embedding': det['embedding'], 'bbox': det['bbox'], 'last_seen': self.frame_idx}
                assigned_tracks.append({'face_id': face_id, 'bbox': det['bbox'], 'embedding': det['embedding']})

        to_delete = []
        for tid, track in self.tracks.items():
            if self.frame_idx - track['last_seen'] > self.max_disappeared:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
            if tid in temporal_buffers:
                del temporal_buffers[tid]

        return assigned_tracks

tracker = SimpleTracker(max_disappeared=30, similarity_threshold=0.6)

def detect_multiple_faces_and_embed(opencv_image):
    face_regions = []

    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    filtered = []
    for (x, y, w, h) in faces:
        ar = w / float(h)
        area = w * h
        if 0.75 < ar < 1.33 and area > 1000:
            filtered.append((x, y, w, h))

    for (x, y, w, h) in filtered:
        padding = 10
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(opencv_image.shape[1], x + w + padding)
        y_end = min(opencv_image.shape[0], y + h + padding)
        face_region = opencv_image[y_start:y_end, x_start:x_end]
        rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

        emb = get_face_embedding(rgb_face)

        _, buffer = cv2.imencode('.jpg', face_region, [cv2.IMWRITE_JPEG_QUALITY, 85])
        face_base64 = base64.b64encode(buffer).decode('utf-8')

        face_regions.append({
            'bbox': [int(x), int(y), int(w), int(h)],
            'face_image': face_base64,
            'embedding': emb,
        })

    return face_regions

async def call_service_for_face(session, service_url, face_data):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            request_data = {"image": face_data['face_image'], "timestamp": time.time()}
            async with session.post(f"{service_url}/predict", json=request_data, timeout=25) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"Service error {service_url}: {response.status} - {error_text[:100]} (attempt {attempt + 1})")
        except asyncio.TimeoutError:
            print(f"Timeout calling {service_url} for face (attempt {attempt + 1})")
        except Exception as e:
            print(f"Error calling {service_url}: {e} (attempt {attempt + 1})")
        if attempt < max_retries - 1:
            await asyncio.sleep(0.5)
    return None

def update_temporal_pattern(face_id, anomaly_detected, anomaly_types, current_time, face_image_base64):
    buffer = temporal_buffers[face_id]
    buffer['anomaly_history'].append(anomaly_detected)
    buffer['total_frames'] += 1
    buffer['face_image_base64'] = face_image_base64
    if anomaly_detected:
        buffer['anomaly_count'] += 1
        for anomaly in anomaly_types:
            buffer['dominant_anomalies'][anomaly] += 1
    buffer['behavioral_timeline'].append({
        'timestamp': current_time,
        'anomaly_detected': anomaly_detected,
        'anomaly_types': anomaly_types.copy() if anomaly_types else []
    })
    alert_threshold = 0.7
    alert_cooldown = 300
    min_frames_for_alert = 50
    if len(buffer['anomaly_history']) >= min_frames_for_alert:
        anomaly_fraction = sum(buffer['anomaly_history']) / len(buffer['anomaly_history'])
        time_since_last_alert = current_time - buffer['last_alert_time']
        if anomaly_fraction >= alert_threshold and time_since_last_alert > alert_cooldown:
            duration_minutes = len(buffer['anomaly_history']) / 60
            dominant_anomalies = [item for item, count in buffer['dominant_anomalies'].most_common(3)]
            alert = TemporalAlert(
                face_id=face_id,
                alert_type="REPEATED_ANOMALY_PATTERN",
                anomaly_pattern=f"Persistent anomalies over {duration_minutes:.1f} minutes",
                duration_minutes=duration_minutes,
                anomaly_percentage=anomaly_fraction * 100,
                dominant_anomalies=dominant_anomalies,
                face_image=buffer['face_image_base64'],
                confidence=anomaly_fraction,
                timestamp=current_time
            )
            buffer['last_alert_time'] = current_time
            print(f"🚨 TEMPORAL ALERT: Face {face_id} - {anomaly_fraction*100:.1f}% anomalies over {duration_minutes:.1f} minutes")
            return alert
    return None

def classify_behavioral_anomaly(emotion_result, drowsiness_result, pose_result):
    
    anomaly_count = 0
    anomaly_types = []

    if emotion_result:
        emotion = emotion_result.get('emotion', '').lower()
        confidence = emotion_result.get('confidence', 0)
        emotion_anomaly = False
        if emotion in ['happy', 'surprise'] and confidence < 0.5:
            emotion_anomaly = True
            anomaly_types.append("low_positive_emotion")
        elif emotion == 'neutral' and confidence < 0.2:
            emotion_anomaly = True
            anomaly_types.append("low_neutral_emotion")
        elif emotion == 'sad' and confidence > 0.5:
            emotion_anomaly = True
            anomaly_types.append("high_sadness")
        elif emotion == 'fear' and confidence > 0.4:
            emotion_anomaly = True
            anomaly_types.append("high_fear")
        elif emotion == 'disgust' and confidence > 0.8:
            emotion_anomaly = True
            anomaly_types.append("high_disgust")
        elif emotion == 'angry' and confidence > 0.6:
            emotion_anomaly = True
            anomaly_types.append("high_anger")
        if emotion_anomaly:
            anomaly_count += 1

    if drowsiness_result:
        level = drowsiness_result.get('drowsiness_level', '').lower()
        confidence = drowsiness_result.get('confidence', 0)
        drowsiness_anomaly = False
        if level == 'alert' and confidence < 0.30:
            drowsiness_anomaly = True
            anomaly_types.append("low_alertness")
        elif level == 'drowsy' and confidence < 0.5:
            drowsiness_anomaly = True
            anomaly_types.append("uncertain_drowsiness")
        elif level == 'very_drowsy' and confidence < 0.4:
            drowsiness_anomaly = True
            anomaly_types.append("uncertain_severe_drowsiness")
        elif level == 'very_drowsy' and confidence >= 0.4:
            drowsiness_anomaly = True
            anomaly_types.append("confirmed_severe_drowsiness")
        if drowsiness_anomaly:
            anomaly_count += 1

    if pose_result:
        confidence = pose_result.get('confidence', 0)
        classification = pose_result.get('pose_classification', '').lower()
        if classification == 'anomalous':
            if confidence > 0.8:
                anomaly_types.append("high_pose_anomaly")
                anomaly_count += 1
            elif confidence > 0.5:
                anomaly_types.append("minor_pose_anomaly")
                anomaly_count += 1

    if anomaly_count == 0:
        return "NORMAL", "NORMAL", "green", anomaly_types, 0.0
    elif anomaly_count == 1:
        return "MINOR_WARNING", "MINOR_WARNING", "yellow", anomaly_types, 0.33
    elif anomaly_count == 2:
        return "WARNING", "WARNING", "darkorange", anomaly_types, 0.66
    else:
        return "HIGH_WARNING", "HIGH_WARNING", "red", anomaly_types, 1.0

def create_face_analysis_summary(face_id, emotion_result, drowsiness_result, pose_result, anomaly_types):
    summary_parts = [f"Face {face_id}:"]
    if emotion_result:
        emotion = emotion_result.get('emotion', 'Unknown')
        emotion_confidence = emotion_result.get('confidence', 0)
        summary_parts.append(f"Emotion: {emotion} ({emotion_confidence:.2f})")
    else:
        summary_parts.append("Emotion: No data")
    if drowsiness_result:
        drowsiness_level = drowsiness_result.get('drowsiness_level', 'Unknown')
        drowsiness_confidence = drowsiness_result.get('confidence', 0)
        summary_parts.append(f"Alertness: {drowsiness_level} ({drowsiness_confidence:.2f})")
    else:
        summary_parts.append("Alertness: No data")
    if pose_result:
        pose_classification = pose_result.get('pose_classification', 'Unknown')
        pose_confidence = pose_result.get('confidence', 0)
        summary_parts.append(f"Pose: {pose_classification} ({pose_confidence:.2f})")
    else:
        summary_parts.append("Pose: No data")
    if anomaly_types:
        summary_parts.append(f"Anomalies: {', '.join(anomaly_types)}")
    return " | ".join(summary_parts)


@app.post("/analyze", response_model=MultiFaceBehavioralResponse)
async def analyze_multiple_faces(request: ImageRequest):
    try:
        start_time = time.time()
        timestamp = request.timestamp or time.time()

        humans_detected, opencv_image = detect_humans_first(request.image)
        if not humans_detected:
            print(" No humans detected - skipping all processing")
            return MultiFaceBehavioralResponse(
                total_faces=0,
                faces_analysis=[],
                overall_scene_status="NO_HUMAN_DETECTED",
                scene_warning_color="gray",
                timestamp=timestamp,
                temporal_alerts=[]
            )

        print("✅ Humans detected - proceeding with full analysis")

        face_detections = detect_multiple_faces_and_embed(opencv_image)
        if not face_detections:
            return MultiFaceBehavioralResponse(
                total_faces=0,
                faces_analysis=[],
                overall_scene_status="HUMANS_DETECTED_NO_FACES",
                scene_warning_color="gray",
                timestamp=timestamp,
                temporal_alerts=[]
            )
        
        tracked_faces = tracker.update(face_detections)

        faces_analysis = []
        temporal_alerts = []

        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for track in tracked_faces:
                face_id = track['face_id']
                bbox = track['bbox']
                face_data = next((f for f in face_detections if f['bbox'] == bbox), None)
                if face_data is None:
                    continue
                
                emotion_task = call_service_for_face(session, EMOTION_SERVICE_URL, face_data)
                drowsiness_task = call_service_for_face(session, DROWSINESS_SERVICE_URL, face_data)
                pose_task = call_service_for_face(session, POSE_SERVICE_URL, face_data)

                emotion_result, drowsiness_result, pose_result = await asyncio.gather(
                    emotion_task, drowsiness_task, pose_task, return_exceptions=True
                )

                if isinstance(emotion_result, Exception):
                    print(f"Emotion service failed for face {face_id}: {emotion_result}")
                    emotion_result = None
                if isinstance(drowsiness_result, Exception):
                    print(f"Drowsiness service failed for face {face_id}: {drowsiness_result}")
                    drowsiness_result = None
                if isinstance(pose_result, Exception):
                    print(f"Pose service failed for face {face_id}: {pose_result}")
                    pose_result = None

                overall_status, warning_level, warning_color, anomaly_types, confidence = classify_behavioral_anomaly(
                    emotion_result, drowsiness_result, pose_result
                )

                anomaly_detected = (overall_status != "NORMAL")
                temporal_alert = update_temporal_pattern(
                    face_id,
                    anomaly_detected,
                    anomaly_types,
                    timestamp,
                    face_data['face_image']
                )

                if temporal_alert:
                    temporal_alerts.append(temporal_alert)

                analysis_summary = create_face_analysis_summary(
                    face_id, emotion_result, drowsiness_result, pose_result, anomaly_types
                )

                face_analysis = FaceAnalysisResult(
                    face_id=face_id,
                    bbox=bbox,
                    emotion_result=emotion_result,
                    drowsiness_result=drowsiness_result,
                    pose_result=pose_result,
                    overall_status=overall_status,
                    warning_level=warning_level,
                    warning_color=warning_color,
                    confidence=confidence,
                    analysis_summary=analysis_summary,
                    anomaly_types=anomaly_types,
                    temporal_alert=temporal_alert
                )

                faces_analysis.append(face_analysis)

        warning_colors = [face.warning_color for face in faces_analysis]
        temporal_alert_count = len(temporal_alerts)

        if temporal_alert_count > 0 or "red" in warning_colors:
            overall_scene_status = "HIGH_WARNING_SCENE"
            scene_warning_color = "red"
        elif "darkorange" in warning_colors:
            overall_scene_status = "WARNING_SCENE"
            scene_warning_color = "darkorange"
        elif "yellow" in warning_colors:
            overall_scene_status = "MINOR_WARNING_SCENE"
            scene_warning_color = "yellow"
        else:
            overall_scene_status = "NORMAL_SCENE"
            scene_warning_color = "green"

        processing_time = time.time() - start_time
        print(f"Multi-face analysis completed in {processing_time:.2f} seconds for {len(tracked_faces)} faces")
        if temporal_alerts:
            print(f"Generated {len(temporal_alerts)} temporal alerts")

        return MultiFaceBehavioralResponse(
            total_faces=len(tracked_faces),
            faces_analysis=faces_analysis,
            overall_scene_status=overall_scene_status,
            scene_warning_color=scene_warning_color,
            timestamp=timestamp,
            temporal_alerts=temporal_alerts
        )

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
@app.get("/temporal_status")
async def get_temporal_status():
    """Get enhanced temporal pattern status for all tracked faces"""
    status = {}
    current_time = time.time()
    
    for face_id, buffer in temporal_buffers.items():
        if buffer['total_frames'] > 0:
            anomaly_rate = buffer['anomaly_count'] / buffer['total_frames']
            time_tracked = (current_time - buffer['creation_time']) / 60
            
            recent_history = list(buffer['anomaly_history'])[-60:]
            recent_anomaly_rate = sum(recent_history) / len(recent_history) if recent_history else 0
            
            status[f"face_{face_id}"] = {
                'total_frames': buffer['total_frames'],
                'anomaly_count': buffer['anomaly_count'],
                'anomaly_rate': anomaly_rate,
                'recent_anomaly_rate': recent_anomaly_rate,
                'time_tracked_minutes': time_tracked,
                'last_alert_time': buffer['last_alert_time'],
                'dominant_anomalies': dict(buffer['dominant_anomalies'].most_common(5)),
                'has_face_image': buffer['face_image_base64'] is not None
            }
    
    return {
        'total_faces_tracked': len(status),
        'faces_status': status,
        'timestamp': current_time,
        'system_uptime_minutes': (current_time - min([b['creation_time'] for b in temporal_buffers.values()])) / 60 if temporal_buffers else 0
    }

@app.get("/clear_temporal_data")
async def clear_temporal_data():
    """Clear all temporal tracking data"""
    global temporal_buffers
    cleared_faces = len(temporal_buffers)
    temporal_buffers.clear()
    
    return {
        "message": f"Cleared temporal data for {cleared_faces} faces",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    global human_detection_model
    
    return {
        "status": "healthy",
        "service": "human_detection_enhanced_behavioral_analysis",
        "version": "2.0.0",
        "human_detection_enabled": human_detection_model is not None,
        "threshold_system": "manual_business_logic",
        "supported_features": [
            "human_detection_prefilter",
            "emotion_detection", 
            "drowsiness_detection", 
            "pose_detection", 
            "multi_face_analysis", 
            "4_level_warnings", 
            "temporal_pattern_analysis",
            "face_image_storage",
            "enhanced_anomaly_classification"
        ],
        "faces_currently_tracked": len(temporal_buffers),
        "temporal_buffer_size": 1200,
        "alert_threshold": 0.7,
        "alert_cooldown_seconds": 300
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
