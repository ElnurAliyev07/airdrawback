from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
from uuid import uuid4
from typing import List, Dict

app = FastAPI(title="AirDraw AI - QuickDraw Smart Format", version="3.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
SAMPLE_IMAGES_DIR = "static/media"  # JPEG dosyalarının bulunduğu dizin
MODEL_PATH = "sketch_recognition_model.h5"
CLASS_NAMES = [
    'airplane', 'apple', 'bicycle', 'bird',
    'butterfly', 'candle', 'car', 'cat',
    'circle', 'clock', 'envelope', 'face',
    'fish', 'flower', 'giraffe', 'house',
    'star', 'sun', 'tree', 'umbrella'
]

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)

# Load QuickDraw model
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ QuickDraw model loaded: {MODEL_PATH}")
    print(f"📊 Model expects: Black background + White strokes")
    print(f"📊 Input shape: {model.input_shape}")
    print(f"📊 Output classes: {model.output_shape}")
except Exception as e:
    print(f"❌ Failed to load QuickDraw model: {str(e)}")
    print("⚠️ Run the QuickDraw training script first")

class ImageData(BaseModel):
    image: str

class PredictionResponse(BaseModel):
    status: str
    message: str
    prediction: dict
    top_5_predictions: List[dict]
    processed_image: str

def preprocess_sketch(image_array):
    """
    Preprocesses frontend sketch to match QuickDraw model input (black bg + white strokes)
    """
    print(f"🔄 Preprocessing sketch for QuickDraw model...")
    print(f"👁️ Input: {image_array.shape}, range: {image_array.min():.2f}-{image_array.max():.2f}")
    
    if len(image_array.shape) == 3:
        image = Image.fromarray(image_array).convert('L')
    else:
        image = Image.fromarray(image_array)
    
    print(f"👁️ After grayscale: mode={image.mode}, size={image.size}")
    
    image = image.resize((28, 28))
    image_array = np.array(image)
    print(f"👁️ After resize: {image_array.shape}")
    
    image_array = image_array.astype('float32') / 255.0
    print(f"👁️ After normalization: range: {image_array.min():.3f}-{image_array.max():.3f}")
    
    mean_val = image_array.mean()
    black_pixels = np.sum(image_array < 0.2)
    white_pixels = np.sum(image_array > 0.8)
    total_pixels = image_array.size
    black_ratio = black_pixels / total_pixels
    white_ratio = white_pixels / total_pixels
    
    print(f"🎯 Format Verification:")
    print(f"   📊 Mean: {mean_val:.3f} (should be LOW for black background)")
    print(f"   📊 Black pixels: {black_ratio:.1%} (background)")
    print(f"   📊 White pixels: {white_ratio:.1%} (strokes)")
    
    if mean_val < 0.5 and black_ratio > 0.6:
        print("   ✅ CORRECT QuickDraw format: Black bg + white strokes")
    else:
        print(f"   ⚠️ Unexpected format (mean={mean_val:.3f}, black={black_ratio:.1%}). May affect predictions.")
    
    final_image = image_array.reshape(1, 28, 28, 1)
    print(f"👁️ Model input shape: {final_image.shape}")
    print(f"✅ Preprocessing completed")
    
    return final_image, image_array

def predict_quickdraw_sketch(image_array):
    """
    Makes predictions using the QuickDraw model
    """
    if model is None:
        raise HTTPException(status_code=500, detail="QuickDraw model not loaded")
    
    try:
        preprocessed_input, debug_image = preprocess_sketch(image_array)
        
        print(f"🤖 Running QuickDraw prediction...")
        predictions = model.predict(preprocessed_input, verbose=0)
        prediction_probs = predictions[0]
        
        top1_index = int(np.argmax(prediction_probs))
        top1_confidence = float(prediction_probs[top1_index] * 100)
        top1_label = CLASS_NAMES[top1_index]
        
        top5_indices = np.argsort(prediction_probs)[-5:][::-1]
        top5_predictions = [
            {
                "rank": i + 1,
                "class_name": CLASS_NAMES[idx],
                "confidence": round(float(prediction_probs[idx] * 100), 2),
                "class_index": int(idx)
            }
            for i, idx in enumerate(top5_indices)
        ]
        
        print(f"🎯 QuickDraw Results:")
        print(f"   🥇 Top-1: {top1_label} ({top1_confidence:.2f}%)")
        print(f"   🏆 Top-5: {[p['class_name'] for p in top5_predictions]}")
        
        all_predictions = {
            CLASS_NAMES[idx]: round(float(prediction_probs[idx] * 100), 2)
            for idx in np.argsort(prediction_probs)[-10:][::-1]
        }
        
        return {
            "class_index": top1_index,
            "class_name": top1_label,
            "confidence": round(top1_confidence, 2),
            "top_5_predictions": top5_predictions,
            "all_predictions": all_predictions,
            "total_classes": len(CLASS_NAMES),
            "model_format": "QuickDraw (Black bg + White strokes)",
            "input_converted": "Minimal preprocessing (matches QuickDraw format)",
            "preprocessing_applied": [
                "Grayscale conversion",
                "Resize to 28x28",
                "Normalization [0,1]"
            ]
        }
        
    except Exception as e:
        print(f"❌ QuickDraw prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"QuickDraw prediction failed: {str(e)}")

def get_quickdraw_sample_image(class_name):
    """
    Returns base64 encoded image (JPG/PNG) for the given QuickDraw class from static/media/
    """
    possible_extensions = ['jpeg', 'jpg', 'png']
    image_path = None

    for ext in possible_extensions:
        candidate_path = os.path.join(SAMPLE_IMAGES_DIR, f"{class_name}.{ext}")
        if os.path.exists(candidate_path):
            image_path = candidate_path
            break

    if not image_path:
        print(f"⚠️ Sample image not found for class '{class_name}' in any format.")
        return f"/placeholder.svg?height=200&width=200&text=❓+{class_name.replace('_', ' ').title()}"

    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"
            return f"data:{mime_type};base64,{encoded_image}"
    except Exception as e:
        print(f"❌ Error reading sample image for '{class_name}': {str(e)}")
        return f"/placeholder.svg?height=200&width=200&text=❓+{class_name.replace('_', ' ').title()}"

@app.get("/")
async def root():
    return {
        "message": "AirDraw AI - QuickDraw Smart Format Backend",
        "version": "3.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "model_info": {
            "classes": len(CLASS_NAMES),
            "format": "QuickDraw (Black background + White strokes)",
            "input_shape": str(model.input_shape) if model else None,
            "output_shape": str(model.output_shape) if model else None,
            "parameters": model.count_params() if model else None
        },
        "frontend_compatibility": {
            "input_format": "Black background + White strokes",
            "conversion": "Minimal preprocessing (matches model format)",
            "output_format": "QuickDraw format (Black bg + White strokes)"
        },
        "preprocessing_pipeline": [
            "1. Grayscale conversion (RGB/RGBA → Gray)",
            "2. Resize to 28x28 pixels",
            "3. Normalization to [0,1]"
        ]
    }

@app.get("/classes")
async def get_classes():
    """Returns QuickDraw classes and categories"""
    categories = {
        "animals": [c for c in CLASS_NAMES if c in ['bird', 'butterfly', 'fish', 'giraffe']],
        "objects": [c for c in CLASS_NAMES if c in ['airplane', 'apple', 'bicycle', 'candle', 'car', 'clock', 'envelope', 'house', 'umbrella']],
        "nature": [c for c in CLASS_NAMES if c in ['flower', 'star', 'sun', 'tree']],
        "shapes": [c for c in CLASS_NAMES if c in ['circle', 'face']]
    }
    
    return {
        "total_classes": len(CLASS_NAMES),
        "all_classes": CLASS_NAMES,
        "categories": categories,
        "model_format": "QuickDraw (Black bg + White strokes)",
        "frontend_format": "Black bg + White strokes (no conversion needed)"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_sketch_endpoint(data: ImageData):
    try:
        print(f"📨 Received sketch prediction request")
        
        # Decode base64 image
        base64_string = data.image
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        
        try:
            image_data = base64.b64decode(base64_string)
            print(f"📨 Decoded image data: {len(image_data)} bytes")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Convert to numpy array
        try:
            image = Image.open(io.BytesIO(image_data))
            print(f"📨 PIL image: mode={image.mode}, size={image.size}")
            
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (0, 0, 0))
                background.paste(image, mask=image.split()[-1])
                image = background
                print("📨 RGBA → RGB with black background")
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"📨 Converted to RGB from {image.mode}")
            
            image_array = np.array(image)
            print(f"📨 Input array: {image_array.shape}, range: {image_array.min()}-{image_array.max()}")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image conversion failed: {str(e)}")
        
        # Save original for debugging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frontend_input_{uuid4().hex[:8]}_{timestamp}.png"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        try:
            with open(filepath, "wb") as f:
                f.write(image_data)
            print(f"💾 Saved input: {filepath}")
        except Exception as e:
            print(f"⚠️ Could not save input: {str(e)}")
        
        # QuickDraw prediction
        print(f"🔄 Starting preprocessing...")
        prediction_result = predict_quickdraw_sketch(image_array)
        
        # Örnek görüntü (JPEG, base64 formatında)
        sample_image_base64 = get_quickdraw_sample_image(prediction_result["class_name"])
        
        print(f"🎯 FINAL RESULT: {prediction_result['class_name']} ({prediction_result['confidence']:.2f}%)")
        
        return PredictionResponse(
            status="success",
            message=f"QuickDraw prediction: '{prediction_result['class_name']}' ({prediction_result['confidence']:.1f}% confidence)",
            prediction=prediction_result,
            top_5_predictions=prediction_result['top_5_predictions'],
            processed_image=sample_image_base64
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sketch processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "QuickDraw CNN",
        "timestamp": datetime.now().isoformat(),
        "classes": len(CLASS_NAMES),
        "frontend_compatibility": {
            "input_expected": "Black background + White strokes",
            "model_expects": "Black background + White strokes",
            "conversion": "Minimal preprocessing"
        },
        "preprocessing_features": [
            "Grayscale conversion",
            "Resize to 28x28",
            "Normalization"
        ]
    }

@app.get("/format-conversion-info")
async def format_conversion_info():
    """Information about preprocessing"""
    return {
        "frontend_format": {
            "background": "Black (0)",
            "strokes": "White (255)",
            "description": "Matches QuickDraw model format"
        },
        "quickdraw_model_format": {
            "background": "Black (0)",
            "strokes": "White (255)",
            "description": "Google QuickDraw dataset format"
        },
        "preprocessing_process": {
            "step_1": "Grayscale conversion (RGB/RGBA → Gray)",
            "step_2": "Resize to 28x28 pixels",
            "step_3": "Normalization to [0,1] for model input",
            "step_4": "Add CNN dimensions (batch + channel)"
        }
    }

@app.get("/debug/{image_name}")
async def debug_preprocessing(image_name: str):
    """Debug preprocessing steps"""
    filepath = os.path.join(UPLOAD_DIR, image_name)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Debug image not found")
    
    try:
        with open(filepath, "rb") as f:
            image_data = f.read()
        
        image = Image.open(io.BytesIO(image_data))
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (0, 0, 0))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        debug_steps = {}
        
        debug_steps["1_original"] = {
            "shape": list(image_array.shape),
            "min": int(image_array.min()),
            "max": int(image_array.max()),
            "mean": float(image_array.mean()),
            "format_detected": "Frontend input (black bg + white strokes)"
        }
        
        if len(image_array.shape) == 3:
            image = Image.fromarray(image_array).convert('L')
        else:
            image = Image.fromarray(image_array)
        
        debug_steps["2_grayscale"] = {
            "shape": list(np.array(image).shape),
            "min": int(np.array(image).min()),
            "max": int(np.array(image).max()),
            "mean": float(np.array(image).mean())
        }
        
        image = image.resize((28, 28))
        resized = np.array(image)
        debug_steps["3_resized"] = {
            "shape": list(resized.shape),
            "min": int(resized.min()),
            "max": int(resized.max()),
            "mean": float(resized.mean())
        }
        
        normalized = resized.astype('float32') / 255.0
        debug_steps["4_normalized"] = {
            "shape": list(normalized.shape),
            "min": float(normalized.min()),
            "max": float(normalized.max()),
            "mean": float(normalized.mean()),
            "ready_for_model": True,
            "format": "QuickDraw (Black bg=0.0, White strokes=1.0)"
        }
        
        return {
            "image_name": image_name,
            "preprocessing_debug": debug_steps,
            "preprocessing_result": "SUCCESS",
            "model_compatibility": "READY"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🎨 Starting AirDraw - QuickDraw Smart Format Backend...")
    print("🌐 Server: http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")
    print("🎯 Model expects: BLACK background + WHITE strokes")
    print("📱 Frontend sends: BLACK background + WHITE strokes")
    print("🔄 Processing: Minimal preprocessing")
    print(f"📊 Supporting {len(CLASS_NAMES)} classes")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)