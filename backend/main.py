import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mnist_backend")

# Add current backend folder to sys.path so preprocess module is always importable
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

try:
    from preprocess import preprocess_canvas_image
except ImportError:
    from backend.preprocess import preprocess_canvas_image

# Global model state
keras_model = None
MODEL_RELATIVE_PATH = os.path.join("model", "mnist_cnn_best.keras")
MODEL_FULL_PATH = os.path.abspath(os.path.join(BACKEND_DIR, MODEL_RELATIVE_PATH))

# Root project directory path
BASE_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load Keras CNN model once when application starts.
    """
    global keras_model
    logger.info(f"Checking for trained model at: {MODEL_FULL_PATH}")
    
    if os.path.exists(MODEL_FULL_PATH):
        try:
            import tensorflow as tf
            logger.info("Loading Keras model...")
            keras_model = tf.keras.models.load_model(MODEL_FULL_PATH)
            logger.info("Successfully loaded mnist_cnn_best.keras model!")
        except Exception as e:
            logger.error(f"Failed to load Keras model: {e}")
            keras_model = None
    else:
        logger.warning(
            f"Model file NOT found at '{MODEL_FULL_PATH}'. "
            "Please place 'mnist_cnn_best.keras' inside the 'backend/model/' directory."
        )
        keras_model = None

    yield
    logger.info("Shutting down backend...")

# Initialize FastAPI application
app = FastAPI(
    title="MNIST Handwritten Digit Classifier API",
    description="Backend API for predicting handwritten digits (0-9) using a trained CNN Keras model.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for all origins and hosts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Pydantic schemas
class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded PNG canvas image")

class PredictResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: dict[str, float]

# =========================================================================
# FRONTEND SERVING ROUTES (Serves entire website directly on Render)
# =========================================================================

@app.get("/")
def serve_index():
    """
    Serve main single-page web app (index.html).
    """
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "MNIST CNN Classifier API is running!",
        "health": "/health",
        "docs_page": "/docs_page",
        "model_loaded": keras_model is not None
    }

@app.get("/docs_page")
def serve_docs_page():
    """
    Serve model documentation page (docs.html).
    """
    docs_path = os.path.join(BASE_DIR, "docs.html")
    if os.path.exists(docs_path):
        return FileResponse(docs_path)
    return {"message": "Documentation page not found"}

@app.get("/styles.css")
def serve_styles():
    styles_path = os.path.join(BASE_DIR, "styles.css")
    if os.path.exists(styles_path):
        return FileResponse(styles_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")

@app.get("/app.js")
def serve_app_js():
    app_js_path = os.path.join(BASE_DIR, "app.js")
    if os.path.exists(app_js_path):
        return FileResponse(app_js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")

# =========================================================================
# BACKEND ML & HEALTH ENDPOINTS
# =========================================================================

@app.get("/health")
def health_check():
    """
    Health check endpoint to confirm backend status and model readiness.
    """
    return {
        "status": "running",
        "model_loaded": keras_model is not None,
        "model_path": MODEL_FULL_PATH,
        "instruction": (
            "Model is active and ready." if keras_model is not None
            else f"Please place 'mnist_cnn_best.keras' at '{MODEL_FULL_PATH}'"
        )
    }

@app.post("/predict", response_model=PredictResponse)
def predict_digit(payload: PredictRequest):
    """
    Prediction endpoint:
    Receives base64 image -> Preprocesses to 28x28 MNIST format -> Predicts probabilities for 0-9.
    Returns HTTP 503 error if model is not loaded.
    """
    if keras_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded on backend server. Ensure mnist_cnn_best.keras is present."
        )

    if not payload.image or len(payload.image.strip()) == 0:
        raise HTTPException(status_code=400, detail="No image data provided.")

    try:
        input_tensor = preprocess_canvas_image(payload.image)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {str(e)}")

    # Run inference strictly using Keras model
    try:
        predictions = keras_model.predict(input_tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        
        probabilities_dict = {
            str(i): round(float(predictions[i]), 6)
            for i in range(10)
        }

        return PredictResponse(
            digit=predicted_digit,
            confidence=round(float(predictions[predicted_digit]), 6),
            probabilities=probabilities_dict
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
