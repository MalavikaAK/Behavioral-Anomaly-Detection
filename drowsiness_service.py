from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from PIL import Image
import io
from pydantic import BaseModel
import os
from ultralytics import YOLO
import torch

# Initialize FastAPI app
app = FastAPI(title="Drowsiness Detection Service", version="2.0.0")

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
drowsiness_labels = ['alert', 'drowsy', 'very_drowsy']

# Pydantic models for request/response
class ImageRequest(BaseModel):
    image: str  # base64 encoded image
    timestamp: float = None

class DrowsinessResponse(BaseModel):
    drowsiness_level: str
    confidence: float
    eye_state: str
    bbox: list = None
    detections: list = None

def load_model():
    """Load the trained YOLOv8 model with CPU compatibility fix"""
    global model
    try:
        # FIXED: Corrected model path
        model_path = os.path.join("..", "..", "shared", "models", r"C:\Users\malu0\OneDrive\Desktop\COLLEGE\behavioral_anomaly\code\shared\models\best.pt")
        
        # Force CPU mode to avoid CUDA issues from Colab training
        print("🔧 Loading YOLOv8 model in CPU mode...")
        model = YOLO(model_path)
        
        # CRITICAL FIX: Force CPU inference to avoid CUDA mismatch
        model.to('cpu')
        
        # Test the model with a dummy prediction to ensure it works and also helps the system from crashing
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        test_results = model.predict(source=dummy_image, verbose=False, device='cpu')
        
        print(f"✅ YOLOv8 model loaded successfully on CPU from {model_path}")
        print(f"📊 Model classes: {list(model.names.values())}")
        
    except Exception as e:
        print(f"❌ Error loading YOLOv8 model: {e}")
        raise e

def preprocess_image(image_data):
    """Enhanced image preprocessing for drowsiness detection"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Enhance image quality for better detection
        # Apply ogram equalization to improve contrast
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.equalizeHist(gray)
        opencv_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        
        return opencv_image
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return None

def analyze_drowsiness(detections):
    """Enhanced drowsiness analysis with better logic"""
    try:
        if not detections or len(detections) == 0:
            return "alert", 0.5, "open"  # Default to alert if no detections
        
        # Get the detection with highest confidence
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        # Map class names to drowsiness levels
        class_name = best_detection['class_name'].lower()
        confidence = best_detection['confidence']
        
        # Enhanced drowsiness classification logic
        if 'closed' in class_name or 'sleep' in class_name:
            if confidence > 0.8:
                drowsiness_level = "very_drowsy"
                eye_state = "closed"
            elif confidence > 0.6:
                drowsiness_level = "drowsy"
                eye_state = "partially_closed"
            else:
                drowsiness_level = "alert"
                eye_state = "open"
        elif 'drowsy' in class_name or 'tired' in class_name:
            if confidence > 0.7:
                drowsiness_level = "drowsy"
                eye_state = "partially_closed"
            else:
                drowsiness_level = "alert"
                eye_state = "open"
        elif 'open' in class_name or 'alert' in class_name or 'awake' in class_name:
            drowsiness_level = "alert"
            eye_state = "open"
        else:
            # Unknown class - default based on confidence
            if confidence > 0.7:
                drowsiness_level = "drowsy"
                eye_state = "partially_closed"
            else:
                drowsiness_level = "alert"
                eye_state = "open"
        
        return drowsiness_level, confidence, eye_state
        
    except Exception as e:
        print(f"❌ Error in drowsiness analysis: {e}")
        return "alert", 0.5, "unknown"

def predict_drowsiness(image):
    """Enhanced drowsiness prediction with CPU-only inference"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # CRITICAL FIX: Force CPU inference to avoid CUDA issues
        results = model.predict(
            source=image, 
            verbose=False,
            device='cpu',        # Force CPU inference
            imgsz=640,          # Standard YOLO input size
            conf=0.25,          # Confidence threshold
            max_det=10          # Maximum detections
        )
        
        detections = []
        bbox = None
        
        # Process results with enhanced error handling
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Validate coordinates
                        if x1 >= x2 or y1 >= y2:
                            continue
                        
                        # Get class name
                        class_name = model.names[class_id] if class_id in model.names else f"class_{class_id}"
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
                        
                        # Use first valid detection for main bbox
                        if bbox is None:
                            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                            
                    except Exception as box_error:
                        print(f"⚠️ Error processing detection box: {box_error}")
                        continue
        
        # Analyze drowsiness level
        drowsiness_level, confidence, eye_state = analyze_drowsiness(detections)
        
        print(f"🔍 Drowsiness prediction: {drowsiness_level} ({confidence:.2f}) - {eye_state}")
        
        return drowsiness_level, confidence, eye_state, bbox, detections
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# FIXED: Load model on startup with proper error handling
@app.on_event("startup")
async def startup_event():
    """Load model on startup with enhanced error handling"""
    try:
        load_model()
        print("🚀 Drowsiness detection service started successfully")
    except Exception as e:
        print(f"💥 Failed to start drowsiness service: {e}")
        # Don't raise exception to allow service to start (for debugging)

# API Endpoints
@app.post("/predict", response_model=DrowsinessResponse)
async def predict_drowsiness_endpoint(request: ImageRequest):
    """Enhanced prediction endpoint with better error handling"""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded - service unavailable")
        
        # Preprocess the image
        image = preprocess_image(request.image)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Predict drowsiness
        drowsiness_level, confidence, eye_state, bbox, detections = predict_drowsiness(image)
        
        # Convert numpy types to Python types to avoid serialization issues
        if bbox:
            bbox = [int(x) for x in bbox]
        
        if detections:
            for detection in detections:
                detection['bbox'] = [int(x) for x in detection['bbox']]
                detection['confidence'] = float(detection['confidence'])
                detection['class_id'] = int(detection['class_id'])
        
        return DrowsinessResponse(
            drowsiness_level=drowsiness_level,
            confidence=float(confidence),
            eye_state=eye_state,
            bbox=bbox,
            detections=detections
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    
    # Test model if loaded
    model_working = False
    if model is not None:
        try:
            # Quick test prediction
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            test_results = model.predict(source=dummy_image, verbose=False, device='cpu')
            model_working = True
        except Exception as e:
            print(f"⚠️ Model health check failed: {e}")
    
    return {
        "status": "healthy" if model_working else "degraded",
        "service": "drowsiness_detection",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "model_working": model_working,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "fixes_applied": ["cpu_forced", "cuda_compatibility", "enhanced_error_handling"]
    }

@app.get("/model_info")
async def model_info():
    """Enhanced model information endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return {
            "model_name": "YOLOv8",
            "model_type": "object_detection",
            "classes": list(model.names.values()),
            "num_classes": len(model.names),
            "input_size": "640x640",
            "framework": "ultralytics",
            "device": "cpu",
            "trained_on": "google_colab_t4_gpu",
            "deployment_mode": "cpu_inference",
            "version": "2.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# Alternative endpoint for file upload
@app.post("/predict_file")
async def predict_drowsiness_file(file: UploadFile = File(...)):
    """Enhanced file upload prediction endpoint"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        contents = await file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Create request object
        request = ImageRequest(image=image_base64)
        
        # Use existing prediction logic
        return await predict_drowsiness_endpoint(request)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.get("/debug/test_prediction")
async def test_prediction():
    """Debug endpoint to test model prediction"""
    try:
        if model is None:
            return {"status": "error", "message": "Model not loaded"}
        
        # Create a test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run prediction
        results = model.predict(source=test_image, verbose=False, device='cpu')
        
        return {
            "status": "success",
            "message": "Model prediction test successful",
            "results_count": len(results),
            "device": "cpu"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Model prediction test failed: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
