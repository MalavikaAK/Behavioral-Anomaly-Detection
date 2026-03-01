from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
import base64
from PIL import Image
import io
from pydantic import BaseModel
import os
import pickle
import mediapipe as mp
from typing import Optional, List

# Initialize FastAPI app
app = FastAPI(title="Pose Detection Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and MediaPipe
lstm_model = None
preprocessor = None
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_labels = ['normal', 'anomalous']

# Pydantic models for request/response
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    timestamp: float = None

class PoseResponse(BaseModel):
    pose_classification: str
    confidence: float
    keypoints: Optional[List[float]] = None
    bbox: Optional[List[int]] = None
    pose_landmarks: Optional[dict] = None


def load_models():
    """Load the trained LSTM model and preprocessor"""
    global lstm_model, preprocessor
    try:
        # FIXED: Use direct paths without redundant concatenation
        model_path = r"C:\Users\malu0\OneDrive\Desktop\COLLEGE\behavioral_anomaly\behavioral_anomaly\shared\models\behavioral_anomaly_model_final.h5"
        lstm_model = tf.keras.models.load_model(model_path)
        print(f"LSTM model loaded successfully from {model_path}")
        
        preprocessor_path = r"C:\Users\malu0\OneDrive\Desktop\COLLEGE\behavioral_anomaly\behavioral_anomaly\shared\models\pose_scaler.pkl"
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded successfully from {preprocessor_path}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

def extract_pose_landmarks(image):
    """Extract pose landmarks using MediaPipe - matching training features (105)"""
    try:
        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe Pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:
            
            # Process the image
            results = pose.process(rgb_image)
            
            if results.pose_landmarks:
                # Extract landmark coordinates (33 landmarks × 3 coordinates = 99 features)
                landmarks = []
                landmark_points = []
                
                for landmark in results.pose_landmarks.landmark:
                    # Only use x, y, z (skip visibility to match training)
                    landmarks.extend([
                        float(landmark.x),
                        float(landmark.y),
                        float(landmark.z)
                    ])
                    # CRITICAL FIX: Convert to float here too
                    landmark_points.append([float(landmark.x), float(landmark.y), float(landmark.z)])
                
                
                
                # Add 6 additional calculated features to reach 105 total
                if len(landmark_points) >= 33:
                    # CRITICAL: Wrap ALL NumPy operations with float()
                    shoulder_dist = float(np.sqrt(
                        (landmark_points[11][0] - landmark_points[12][0])**2 +
                        (landmark_points[11][1] - landmark_points[12][1])**2 +
                        (landmark_points[11][2] - landmark_points[12][2])**2
                    ))
                    
                    hip_dist = float(np.sqrt(
                        (landmark_points[23][0] - landmark_points[24][0])**2 +
                        (landmark_points[23][1] - landmark_points[24][1])**2 +
                        (landmark_points[23][2] - landmark_points[24][2])**2
                    ))
                    
                    head_y = landmark_points[0][1]
                    shoulder_center_y = (landmark_points[11][1] + landmark_points[12][1]) / 2
                    head_shoulder_dist = float(abs(head_y - shoulder_center_y))
                    
                    hip_center_y = (landmark_points[23][1] + landmark_points[24][1]) / 2
                    torso_length = float(abs(shoulder_center_y - hip_center_y))
                    
                    arm_span = shoulder_dist
                    
                    # CRITICAL: Wrap np.mean() with float()
                    left_side_avg = float(np.mean([landmark_points[11][0], landmark_points[23][0]]))
                    right_side_avg = float(np.mean([landmark_points[12][0], landmark_points[24][0]]))
                    symmetry_measure = float(abs(left_side_avg - right_side_avg))
                    
                    # Add the 6 calculated features (99 + 6 = 105)
                    landmarks.extend([
                        shoulder_dist,
                        hip_dist,
                        head_shoulder_dist,
                        torso_length,
                        arm_span,
                        symmetry_measure
                    ])
                
                # Ensure exactly 105 features
                if len(landmarks) != 105:
                    print(f"Warning: Expected 105 features, got {len(landmarks)}")
                    while len(landmarks) < 105:
                        landmarks.append(0.0)
                    landmarks = landmarks[:105]
                
                # Convert to numpy array with explicit dtype
                landmarks_array = np.array(landmarks, dtype=float).reshape(1, -1)
                
                # CRITICAL: Ensure ALL values are Python types
                pose_landmarks_dict = {
                    'landmarks': [
                        {
                            'x': float(landmark.x),
                            'y': float(landmark.y),
                            'z': float(landmark.z),
                            'visibility': float(landmark.visibility)
                        }
                        for landmark in results.pose_landmarks.landmark
                    ]
                }
                
                print(f"Extracted {landmarks_array.shape[1]} features for pose analysis")
                return landmarks_array, pose_landmarks_dict
            else:
                return None, None
                
    except Exception as e:
        print(f"Error in pose landmark extraction: {e}")
        return None, None

def preprocess_image(image_data):
    """Preprocess image for pose detection"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def predict_pose(landmarks_array):
    """Predict pose classification using LSTM model"""
    try:
        if lstm_model is None or preprocessor is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Preprocess landmarks using the loaded preprocessor
        if hasattr(preprocessor, 'transform'):
            processed_landmarks = preprocessor.transform(landmarks_array)
        else:
            processed_landmarks = landmarks_array
        
        # Reshape for LSTM input (assuming sequence length of 1)
        lstm_input = processed_landmarks.reshape(1, 1, -1)
        
        # Make prediction
        predictions = lstm_model.predict(lstm_input)
        
        # CRITICAL FIX: Convert ALL NumPy types to Python types
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        pose_classification = pose_labels[predicted_class]
        
        return pose_classification, confidence
        
    except Exception as e:
        print(f"Error in pose prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

# API Endpoints
@app.post("/predict", response_model=PoseResponse)
async def predict_pose_endpoint(request: ImageRequest):
    """Main prediction endpoint"""
    try:
        # Preprocess the image
        image = preprocess_image(request.image)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Extract pose landmarks
        landmarks_array, pose_landmarks_dict = extract_pose_landmarks(image)
        
        if landmarks_array is None:
            raise HTTPException(status_code=400, detail="No pose detected in image")
        
        # Predict pose classification
        pose_classification, confidence = predict_pose(landmarks_array)
        
        # CRITICAL FIX: Ensure ALL values are Python types
        keypoints = [float(x) for x in landmarks_array.flatten().tolist()] if landmarks_array is not None else []
        
        return PoseResponse(
            pose_classification=str(pose_classification),
            confidence=float(confidence),
            keypoints=keypoints,
            bbox=None,
            pose_landmarks=pose_landmarks_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pose_detection",
        "lstm_model_loaded": lstm_model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if lstm_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": "LSTM Pose Classifier",
        "input_shape": lstm_model.input_shape,
        "output_shape": lstm_model.output_shape,
        "pose_labels": pose_labels,
        "total_parameters": int(lstm_model.count_params()),  # Convert to int
        "mediapipe_version": mp.__version__
    }

@app.post("/predict_file")
async def predict_pose_file(file: UploadFile = File(...)):
    """Predict pose from uploaded file"""
    try:
        # Read file content
        contents = await file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Create request object
        request = ImageRequest(image=image_base64)
        
        # Use existing prediction logic
        return await predict_pose_endpoint(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)