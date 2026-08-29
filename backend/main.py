import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mnist_backend")

# Global model state
keras_model = None
MODEL_RELATIVE_PATH = os.path.join("model", "mnist_cnn_best.keras")
MODEL_FULL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_RELATIVE_PATH))

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

# Robust CORS Configuration for Vercel, Localhost, and All Origins
# Note: allow_credentials must be False when allow_origins=["*"] is enabled to adhere to browser CORS standards.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://number-cnn.vercel.app",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
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

@app.get("/")
def root():
    """
    Root API endpoint welcome message.
    """
    return {
        "message": "MNIST CNN Classifier Backend API is running!",
        "health": "/health",
        "docs": "/docs",
        "model_loaded": keras_model is not None
    }

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

    from preprocess import preprocess_canvas_image

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
