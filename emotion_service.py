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
from contextlib import asynccontextmanager

# Initialize FastAPI app
app = FastAPI(title="Emotion Detection Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Pydantic models for request/response
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    timestamp: float = None

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    bbox: list = None
    all_predictions: dict = None

def load_model():
    """Load the trained Mini-Xception model"""
    global model
    try:
        model_path = os.path.join("..", "..", "shared", "models", r"C:\Users\malu0\OneDrive\Desktop\COLLEGE\behavioral_anomaly\behavioral_anomaly\shared\models\10-06-best_mini_xception_paper.h5")
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_image(image_data):
    """Preprocess image for emotion detection"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect face using Haar cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the first face
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to 48x48 for Mini-Xception
            face_roi = cv2.resize(face_roi, (48, 48))
            
            # Normalize pixel values
            face_roi = face_roi.astype('float32') / 255.0
            
            # Reshape for model input
            face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
            face_roi = np.expand_dims(face_roi, axis=0)   # Add batch dimension
            
            # Convert bbox coordinates to Python int (fix for serialization)
            bbox = [int(x), int(y), int(w), int(h)]
            
            return face_roi, bbox
        else:
            return None, None
            
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None

def predict_emotion(preprocessed_image):
    """Predict emotion using the loaded model"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Make prediction
        predictions = model.predict(preprocessed_image)
        
        # Convert NumPy types to Python native types (CRITICAL FIX)
        predicted_class = int(np.argmax(predictions[0]))  # Convert numpy.int64 to int
        confidence = float(predictions[0][predicted_class])  # Convert numpy.float32 to float
        emotion = emotion_labels[predicted_class]
        
        # Create dictionary of all predictions - convert ALL NumPy values to Python types
        all_predictions = {
            emotion_labels[i]: float(predictions[0][i])  # Convert each numpy.float32 to float
            for i in range(len(emotion_labels))
        }
        
        return emotion, confidence, all_predictions
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Application startup: Loading model...")
    load_model()
    yield
    # Code to run on shutdown (if any)
    print("Application shutdown.")

# Initialize FastAPI app with the new lifespan manager
app = FastAPI(
    title="Emotion Detection Service",
    version="1.0.0",
    lifespan=lifespan
)

# API Endpoints
@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion_endpoint(request: ImageRequest):
    """Main prediction endpoint"""
    try:
        # Preprocess the image
        preprocessed_image, bbox = preprocess_image(request.image)
        
        if preprocessed_image is None:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Predict emotion
        emotion, confidence, all_predictions = predict_emotion(preprocessed_image)
        
        return EmotionResponse(
            emotion=emotion,
            confidence=confidence,
            bbox=bbox,
            all_predictions=all_predictions
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
        "service": "emotion_detection",
        "model_loaded": model is not None
    }

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_name": "Mini-Xception",
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "emotion_labels": emotion_labels,
        "total_parameters": int(model.count_params())  # Convert to int for serialization
    }

# Alternative endpoint for file upload
@app.post("/predict_file")
async def predict_emotion_file(file: UploadFile = File(...)):
    """Predict emotion from uploaded file"""
    try:
        # Read file content
        contents = await file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Create request object
        request = ImageRequest(image=image_base64)
        
        # Use existing prediction logic
        return await predict_emotion_endpoint(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)