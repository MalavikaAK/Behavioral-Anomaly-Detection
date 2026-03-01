import streamlit as st
import cv2
import requests
import base64
import numpy as np
import time
import json
from PIL import Image
import io
import threading
from datetime import datetime
from threading import Thread

# Configuration
BEHAVIORAL_ANALYSIS_URL = "http://localhost:8005"
EMOTION_SERVICE_URL = "http://localhost:8001"
DROWSINESS_SERVICE_URL = "http://localhost:8002"
POSE_SERVICE_URL = "http://localhost:8004"

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'temporal_alerts' not in st.session_state:
    st.session_state.temporal_alerts = []

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #ff9a9e, #fecfef);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .temporal-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        animation: pulse 2s infinite;
        border-left: 5px solid #ff4757;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .status-card {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .warning-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .normal { background: #00b894; color: white; }
    .minor-warning { background: #fdcb6e; color: #2d3436; }
    .warning { background: #e17055; color: white; }
    .high-warning { background: #d63031; color: white; }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

class ThreadedCamera:
    """Threaded camera class to reduce lag"""
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.FPS = 1/30
        self.ret, self.frame = self.capture.read()
        self.stopped = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def update(self):
        while True:
            if self.stopped:
                return
            
            if self.capture.isOpened():
                for _ in range(2):
                    self.capture.grab()
                self.ret, self.frame = self.capture.read()
            
            time.sleep(self.FPS)
    
    def read(self):
        return self.ret, self.frame
    
    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 with optimization"""
    try:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

def get_warning_color_rgb(color_name):
    """Convert color names to RGB values for OpenCV"""
    color_map = {
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'darkorange': (0, 165, 255),
        'orange': (0, 165, 255),
        'red': (0, 0, 255),
        'gray': (128, 128, 128)
    }
    return color_map.get(color_name, (255, 255, 255))

def draw_multi_face_annotations(frame, analysis_result=None):
    """Draw bounding boxes with temporal alerts and enhanced styling"""
    annotated_frame = frame.copy()
    
    if not analysis_result or not analysis_result.get('faces_analysis'):
        return annotated_frame
    
    for face_analysis in analysis_result['faces_analysis']:
        bbox = face_analysis['bbox']
        x, y, w, h = bbox
        
        # Get warning color and temporal alert
        warning_color_name = face_analysis.get('warning_color', 'green')
        color = get_warning_color_rgb(warning_color_name)
        temporal_alert = face_analysis.get('temporal_alert')
        
        # Get analysis data
        emotion_text = "Unknown"
        emotion_conf = 0.0
        if face_analysis.get('emotion_result'):
            emotion_data = face_analysis['emotion_result']
            emotion_text = emotion_data.get('emotion', 'Unknown')
            emotion_conf = emotion_data.get('confidence', 0.0)
        
        drowsiness_text = "Unknown"
        drowsiness_conf = 0.0
        if face_analysis.get('drowsiness_result'):
            drowsiness_data = face_analysis['drowsiness_result']
            drowsiness_text = drowsiness_data.get('drowsiness_level', 'Unknown')
            drowsiness_conf = drowsiness_data.get('confidence', 0.0)
        
        pose_text = "Unknown"
        pose_conf = 0.0
        if face_analysis.get('pose_result'):
            pose_data = face_analysis['pose_result']
            pose_text = pose_data.get('pose_classification', 'Unknown')
            pose_conf = pose_data.get('confidence', 0.0)
        
        warning_level = face_analysis.get('warning_level', 'NORMAL')
        anomaly_types = face_analysis.get('anomaly_types', [])
        
        # Enhanced bounding box with temporal alert styling
        thickness = 5 if temporal_alert else 3
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
        
        # Add temporal alert indicator
        if temporal_alert:
            # Draw pulsing border for temporal alerts
            cv2.rectangle(annotated_frame, (x-8, y-8), (x + w + 8, y + h + 8), (0, 0, 255), 3)
            cv2.putText(annotated_frame, "⚠ TEMPORAL ALERT ⚠", (x, y - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"{temporal_alert['anomaly_percentage']:.1f}% anomalies", (x, y - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Enhanced labels
        labels = [
            f"Face {face_analysis['face_id']} - {warning_level}",
            f"Emotion: {emotion_text} ({emotion_conf:.2f})",
            f"Alert: {drowsiness_text} ({drowsiness_conf:.2f})",
            f"Pose: {pose_text} ({pose_conf:.2f})"
        ]
        
        if anomaly_types:
            labels.append(f"Anomalies: {', '.join(anomaly_types[:2])}")
        
        # Draw enhanced background
        label_height = 28
        total_height = len(labels) * label_height
        cv2.rectangle(annotated_frame, 
                     (x, y - total_height - 15), 
                     (x + max(350, w), y), 
                     (0, 0, 0), -1)
        
        # Add gradient effect
        cv2.rectangle(annotated_frame, 
                     (x, y - total_height - 15), 
                     (x + max(350, w), y - total_height - 10), 
                     color, -1)
        
        # Draw labels with enhanced styling
        for j, label in enumerate(labels):
            label_y = y - total_height + (j * label_height) + 20
            
            if j == 0:  # Face ID and warning level
                text_color = (255, 255, 255)
                font_scale = 0.7
                thickness = 2
            elif j == 4:  # Anomaly types
                text_color = (0, 100, 255)
                font_scale = 0.5
                thickness = 1
            else:  # Regular info
                text_color = (255, 255, 255)
                font_scale = 0.6
                thickness = 1
            
            cv2.putText(annotated_frame, label, (x + 8, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    # Enhanced scene status indicator
    scene_color_name = analysis_result.get('scene_warning_color', 'green')
    scene_color = get_warning_color_rgb(scene_color_name)
    scene_status = analysis_result.get('overall_scene_status', 'NORMAL')
    
    cv2.rectangle(annotated_frame, (10, 10), (350, 60), scene_color, -1)
    cv2.rectangle(annotated_frame, (10, 10), (350, 60), (0, 0, 0), 3)
    cv2.putText(annotated_frame, f"SCENE: {scene_status}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return annotated_frame

def call_multi_face_behavioral_analysis(image_base64):
    """Call the updated multi-face behavioral analysis service"""
    try:
        payload = {
            "image": image_base64,
            "timestamp": time.time()
        }
        
        response = requests.post(
            f"{BEHAVIORAL_ANALYSIS_URL}/analyze",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def get_temporal_status():
    """Get temporal tracking status from the service"""
    try:
        response = requests.get(f"{BEHAVIORAL_ANALYSIS_URL}/temporal_status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def display_temporal_alerts(temporal_alerts):
    """Display temporal alerts with beautiful styling"""
    if not temporal_alerts:
        return
    
    st.markdown("### 🚨 **Temporal Anomaly Alerts**")
    
    for alert in temporal_alerts:
        st.markdown(f"""
        <div class="temporal-alert">
            <h3>🚨 TEMPORAL ALERT: Face {alert['face_id']}</h3>
            <p><strong>Pattern:</strong> {alert['anomaly_pattern']}</p>
            <p><strong>Duration:</strong> {alert['duration_minutes']:.1f} minutes</p>
            <p><strong>Anomaly Rate:</strong> {alert['anomaly_percentage']:.1f}%</p>
            <p><strong>Dominant Issues:</strong> {', '.join(alert['dominant_anomalies'])}</p>
            <p><strong>Confidence:</strong> {alert['confidence']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

def display_temporal_status():
    """Display temporal tracking status"""
    temporal_status = get_temporal_status()
    
    if not temporal_status or not temporal_status.get('faces_status'):
        st.info("🔍 No faces currently being tracked for temporal patterns")
        return
    
    st.markdown("### 📊 **Temporal Pattern Tracking**")
    
    faces_status = temporal_status['faces_status']
    
    cols = st.columns(min(3, len(faces_status)))
    
    for i, (face_id, status) in enumerate(faces_status.items()):
        with cols[i % 3]:
            anomaly_rate = status['anomaly_rate']
            
            if anomaly_rate > 0.7:
                card_class = "high-warning"
                status_emoji = "🔴"
            elif anomaly_rate > 0.4:
                card_class = "warning"
                status_emoji = "🟠"
            elif anomaly_rate > 0.2:
                card_class = "minor-warning"
                status_emoji = "🟡"
            else:
                card_class = "normal"
                status_emoji = "🟢"
            
            st.markdown(f"""
            <div class="status-card">
                <h4>{status_emoji} {face_id}</h4>
                <p><strong>Tracked:</strong> {status['time_tracked_minutes']:.1f} min</p>
                <p><strong>Frames:</strong> {status['total_frames']}</p>
                <p><strong>Anomaly Rate:</strong> {anomaly_rate*100:.1f}%</p>
                <p><strong>Dominant Issues:</strong> {', '.join(list(status['dominant_anomalies'].keys())[:2])}</p>
            </div>
            """, unsafe_allow_html=True)

def display_multi_face_analysis_results(result):
    """Display multi-face analysis results with enhanced UI"""
    if not result:
        return
    
    # Enhanced scene header
    scene_color = result.get('scene_warning_color', 'green')
    scene_status = result.get('overall_scene_status', 'NORMAL')
    
    streamlit_colors = {
        'green': ':green[🟢 NORMAL]',
        'yellow': ':orange[🟡 MINOR WARNING]',
        'darkorange': ':red[🟠 WARNING]',
        'red': ':red[🔴 HIGH WARNING]'
    }
    
    st.markdown(f"""
    <div class="main-header">
        <h2>📊 Scene Analysis - {result['total_faces']} Face(s) Detected</h2>
        <h3>{streamlit_colors.get(scene_color, scene_status)}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display temporal alerts first
    temporal_alerts = result.get('temporal_alerts', [])
    if temporal_alerts:
        display_temporal_alerts(temporal_alerts)
        st.session_state.temporal_alerts.extend(temporal_alerts)
    
    # Enhanced metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Faces", result['total_faces'])
    with col2:
        st.metric("🚨 Active Alerts", len(temporal_alerts))
    #with col3:
        #st.metric("⏱️ Analysis Time", f"{datetime.now().strftime('%H:%M:%S')}")
    
    # Individual face analysis
    if result.get('faces_analysis'):
        st.markdown("### 👥 **Individual Face Analysis**")
        
        if len(result['faces_analysis']) > 1:
            tab_names = []
            for face in result['faces_analysis']:
                face_id = face['face_id']
                warning_level = face.get('warning_level', 'NORMAL')
                temporal_indicator = " ⚠️" if face.get('temporal_alert') else ""
                tab_names.append(f"Face {face_id} ({warning_level}){temporal_indicator}")
            
            face_tabs = st.tabs(tab_names)
            
            for i, face_analysis in enumerate(result['faces_analysis']):
                with face_tabs[i]:
                    display_single_face_analysis(face_analysis)
        else:
            display_single_face_analysis(result['faces_analysis'][0])

def display_single_face_analysis(face_analysis):
    """Display enhanced analysis for a single face"""
    warning_level = face_analysis.get('warning_level', 'NORMAL')
    warning_color = face_analysis.get('warning_color', 'green')
    anomaly_types = face_analysis.get('anomaly_types', [])
    temporal_alert = face_analysis.get('temporal_alert')
    
    # Enhanced status display
    status_classes = {
        "NORMAL": "normal",
        "MINOR_WARNING": "minor-warning", 
        "WARNING": "warning",
        "HIGH_WARNING": "high-warning"
    }
    
    status_emojis = {
        "NORMAL": "🟢",
        "MINOR_WARNING": "🟡", 
        "WARNING": "🟠",
        "HIGH_WARNING": "🔴"
    }
    
    # Main metrics with beautiful cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Face ID</h4>
            <h2>{face_analysis['face_id']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        emoji = status_emojis.get(warning_level, "⚪")
        st.markdown(f"""
        <div class="metric-card">
            <h4>Warning Level</h4>
            <h2>{emoji} {warning_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Confidence</h4>
            <h2>{face_analysis['confidence']:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Anomalies</h4>
            <h2>{len(anomaly_types)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Temporal alert section
    if temporal_alert:
        st.markdown("#### ⚠️ **Temporal Alert Active**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Duration", f"{temporal_alert['duration_minutes']:.1f} min")
        with col2:
            st.metric("📊 Anomaly Rate", f"{temporal_alert['anomaly_percentage']:.1f}%")
        with col3:
            st.metric("🎯 Confidence", f"{temporal_alert['confidence']:.2f}")
    
    # Enhanced anomaly display
    if anomaly_types:
        st.markdown("#### ⚠️ **Detected Anomalies**")
        anomaly_descriptions = {
            'low_positive_emotion': 'Low confidence in positive emotions',
            'low_neutral_emotion': 'Low confidence in neutral expression',
            'high_sadness': 'High sadness detected',
            'high_fear': 'High fear detected',
            'high_disgust': 'High disgust detected',
            'low_alertness': 'Low alertness confidence',
            'uncertain_drowsiness': 'Uncertain drowsiness state',
            'uncertain_severe_drowsiness': 'Uncertain severe drowsiness',
            'minor_pose_anomaly': 'Minor pose anomaly',
            'high_pose_anomaly': 'High confidence pose anomaly'
        }
        
        for anomaly in anomaly_types:
            description = anomaly_descriptions.get(anomaly, anomaly)
            if 'high' in anomaly or 'severe' in anomaly:
                st.error(f"🔴 {description}")
            elif 'low' in anomaly or 'uncertain' in anomaly:
                st.warning(f"🟡 {description}")
            else:
                st.info(f"🔵 {description}")
    
    # Enhanced detailed analysis
    st.markdown("#### 📊 **Detailed Analysis**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎭 Emotion Analysis**")
        if face_analysis.get('emotion_result'):
            emotion_data = face_analysis['emotion_result']
            emotion = emotion_data.get('emotion', 'Unknown')
            confidence = emotion_data.get('confidence', 0)
            
            emotion_colors = {
                'happy': '🟢', 'surprise': '🟡', 'neutral': '⚪',
                'sad': '🔵', 'fear': '🟠', 'disgust': '🔴', 'angry': '🔴'
            }
            emoji = emotion_colors.get(emotion.lower(), '⚪')
            
            st.markdown(f"{emoji} **{emotion}** ({confidence:.2f})")
        else:
            st.write("No emotion data available")
    
    with col2:
        st.markdown("**😴 Drowsiness Analysis**")
        if face_analysis.get('drowsiness_result'):
            drowsiness_data = face_analysis['drowsiness_result']
            level = drowsiness_data.get('drowsiness_level', 'Unknown')
            confidence = drowsiness_data.get('confidence', 0)
            
            drowsiness_colors = {
                'alert': '🟢', 'drowsy': '🟡', 'very_drowsy': '🔴'
            }
            emoji = drowsiness_colors.get(level.lower(), '⚪')
            
            st.markdown(f"{emoji} **{level}** ({confidence:.2f})")
        else:
            st.write("No drowsiness data available")
    
    with col3:
        st.markdown("**🤸 Pose Analysis**")
        if face_analysis.get('pose_result'):
            pose_data = face_analysis['pose_result']
            classification = pose_data.get('pose_classification', 'Unknown')
            confidence = pose_data.get('confidence', 0)
            
            pose_colors = {
                'normal': '🟢', 'anomalous': '🔴'
            }
            emoji = pose_colors.get(classification.lower(), '⚪')
            
            st.markdown(f"{emoji} **{classification}** ({confidence:.2f})")
        else:
            st.write("No pose data available")
    
    # Enhanced summary
    st.markdown("#### 📋 **Analysis Summary**")
    st.info(face_analysis.get('analysis_summary', 'No summary available'))

def check_services_health():
    """Check if all services are running"""
    services = {
        "Behavioral Analysis": BEHAVIORAL_ANALYSIS_URL,
        "Emotion Detection": EMOTION_SERVICE_URL,
        "Drowsiness Detection": DROWSINESS_SERVICE_URL,
        "Pose Detection": POSE_SERVICE_URL
    }
    
    health_status = {}
    
    for service_name, url in services.items():
        try:
            response = requests.get(f"{url}/health", timeout=5)
            health_status[service_name] = response.status_code == 200
        except:
            health_status[service_name] = False
    
    return health_status

def main():
    st.set_page_config(
        page_title="Behavioral Anomaly Detection with Temporal Analysis",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Beautiful main header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Behavioral Anomaly Detection System</h1>
        <p>Advanced AI-Powered Real-time Analysis with Temporal Pattern Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 System Controls")
        
        # Service health check
        st.subheader("🏥 Service Health")
        health_status = check_services_health()
        
        for service, is_healthy in health_status.items():
            status_icon = "✅" if is_healthy else "❌"
            status_color = "🟢" if is_healthy else "🔴"
            st.markdown(f"{status_icon} {status_color} **{service}**")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Camera controls
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📹 Camera Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Start Camera", use_container_width=True):
                st.session_state.camera_active = True
        with col2:
            if st.button("⏹️ Stop Camera", use_container_width=True):
                st.session_state.camera_active = False
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Settings
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("⚙️ Settings")
        analysis_interval = st.slider("Analysis Interval (seconds)", 3, 10, 5)
        show_bounding_boxes = st.checkbox("Show Enhanced Annotations", value=True)
        show_temporal_data = st.checkbox("Show Temporal Tracking", value=True)
        show_anomaly_details = st.checkbox("Show Anomaly Details", value=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced threshold information
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("📊 Detection Thresholds")
        
        with st.expander("🎭 Emotion Thresholds"):
            st.markdown("""
            - **Happy/Surprise**: < 0.5 confidence
            - **Neutral**: < 0.2 confidence  
            - **Sad**: > 0.5 confidence
            - **Fear**: > 0.4 confidence
            - **Disgust**: > 0.8 confidence
            """)
        
        with st.expander("😴 Drowsiness Thresholds"):
            st.markdown("""
            - **Alert**: < 0.3 confidence
            - **Drowsy**: < 0.5 confidence
            - **Very Drowsy**: < 0.4 confidence
            """)
        
        with st.expander("🤸 Pose Thresholds"):
            st.markdown("""
            - **Minor Anomaly**: > 0.5 confidence
            - **High Anomaly**: > 0.8 confidence
            """)
        
        with st.expander("⏱️ Temporal Analysis"):
            st.markdown("""
            - **Alert Threshold**: 70% anomalies in window
            - **Cooldown Period**: 5 minutes between alerts
            - **Buffer Size**: 1200 frames (20 minutes)
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    if not all(health_status.values()):
        st.error("⚠️ Some services are offline. Please start all services before proceeding.")
        return
    
    # Display temporal tracking status
    if show_temporal_data:
        display_temporal_status()
        st.markdown("---")
    
    # Camera section
    if st.session_state.camera_active:
        st.markdown("### 📹 **Live Analysis**")
        
        video_placeholder = st.empty()
        results_placeholder = st.empty()
        
        camera = ThreadedCamera(0)
        
        if not camera.capture.isOpened():
            st.error("❌ Cannot access camera. Please check your camera connection.")
            return
        
        last_analysis_time = 0
        
        try:
            while st.session_state.camera_active:
                ret, frame = camera.read()
                
                if not ret:
                    st.error("❌ Failed to capture frame from camera")
                    break
                
                if show_bounding_boxes:
                    annotated_frame = draw_multi_face_annotations(frame, st.session_state.current_analysis)
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                current_time = time.time()
                if current_time - last_analysis_time >= analysis_interval:
                    
                    image_base64 = encode_image_to_base64(frame)
                    
                    if image_base64:
                        with st.spinner("🔍 Analyzing behavioral patterns..."):
                            result = call_multi_face_behavioral_analysis(image_base64)
                        
                        if result:
                            st.session_state.current_analysis = result
                            
                            with results_placeholder.container():
                                display_multi_face_analysis_results(result)
                            
                            result['timestamp_readable'] = datetime.now().strftime("%H:%M:%S")
                            st.session_state.analysis_history.append(result)
                            
                            if len(st.session_state.analysis_history) > 10:
                                st.session_state.analysis_history.pop(0)
                    
                    last_analysis_time = current_time
                
                time.sleep(0.03)
                
        except Exception as e:
            st.error(f"❌ Camera error: {e}")
        finally:
            camera.stop()
    
    else:
        st.markdown("### 📸 **Image Analysis**")
        uploaded_file = st.file_uploader("Choose an image for analysis...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            if st.button("🔍 **Analyze Behavioral Patterns**", use_container_width=True):
                with st.spinner("🧠 AI is analyzing behavioral patterns..."):
                    result = call_multi_face_behavioral_analysis(image_base64)
                
                if result:
                    display_multi_face_analysis_results(result)
                    
                    if show_bounding_boxes and result.get('faces_analysis'):
                        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        annotated_image = draw_multi_face_annotations(opencv_image, result)
                        annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption="🎯 Enhanced Analysis Results", use_container_width=True)
    
    # Enhanced analysis history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### 📈 **Analysis History**")
        
        # Show recent temporal alerts
        if st.session_state.temporal_alerts:
            with st.expander(f"🚨 Recent Temporal Alerts ({len(st.session_state.temporal_alerts)})"):
                for alert in st.session_state.temporal_alerts[-5:]:
                    st.markdown(f"""
                    **Face {alert['face_id']}** - {alert['anomaly_pattern']} 
                    ({alert['anomaly_percentage']:.1f}% anomalies over {alert['duration_minutes']:.1f} min)
                    """)
        
        for i, result in enumerate(reversed(st.session_state.analysis_history[-5:]), 1):
            scene_color = result.get('scene_warning_color', 'green')
            color_emoji = {'green': '🟢', 'yellow': '🟡', 'darkorange': '🟠', 'red': '🔴'}.get(scene_color, '⚪')
            
            with st.expander(f"{color_emoji} Analysis {i} - {result.get('timestamp_readable', 'Unknown')} - {result.get('total_faces', 0)} face(s)"):
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Faces", result.get('total_faces', 0))
                with col2:
                    st.metric("Scene Status", result.get('overall_scene_status', 'Unknown'))
                with col3:
                    alerts_count = len(result.get('temporal_alerts', []))
                    st.metric("Temporal Alerts", alerts_count)
                
                if result.get('faces_analysis'):
                    st.markdown("**Individual Face Analysis:**")
                    for face_analysis in result['faces_analysis']:
                        face_id = face_analysis['face_id']
                        warning_level = face_analysis.get('warning_level', 'NORMAL')
                        warning_color = face_analysis.get('warning_color', 'green')
                        anomaly_types = face_analysis.get('anomaly_types', [])
                        summary = face_analysis.get('analysis_summary', 'No summary')
                        temporal_alert = face_analysis.get('temporal_alert')
                        
                        face_emoji = {'green': '🟢', 'yellow': '🟡', 'darkorange': '🟠', 'red': '🔴'}.get(warning_color, '⚪')
                        alert_indicator = " ⚠️" if temporal_alert else ""
                        
                        st.write(f"{face_emoji} **Face {face_id} ({warning_level}){alert_indicator}:** {summary}")
                        
                        if anomaly_types and show_anomaly_details:
                            st.write(f"   └─ Anomalies: {', '.join(anomaly_types)}")
                        
                        if temporal_alert and show_anomaly_details:
                            st.write(f"   └─ Alert: {temporal_alert['anomaly_percentage']:.1f}% anomalies over {temporal_alert['duration_minutes']:.1f} min")

if __name__ == "__main__":
    main()
