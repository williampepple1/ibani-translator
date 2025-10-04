"""
FastAPI server for English to Ibani translation.
Supports both rule-based and model-based translation approaches.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import os
from rule_based_translator import IbaniRuleBasedTranslator
from huggingface_translator import IbaniHuggingFaceTranslator


# Initialize FastAPI app
app = FastAPI(
    title="Ibani Translator API",
    description="English to Ibani translation service using both rule-based and model-based approaches",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize translators
rule_based_translator = None
model_translator = None


class TranslationRequest(BaseModel):
    text: str
    tense: str = "present"
    method: str = "rule_based"  # "rule_based" or "model"


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    method: str = "rule_based"
    confidence: Optional[float] = None


class BatchTranslationRequest(BaseModel):
    texts: List[str]
    tense: str = "present"
    method: str = "rule_based"  # "rule_based" or "model"


class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]


@app.on_event("startup")
async def startup_event():
    """Initialize translators on startup."""
    global rule_based_translator, model_translator
    
    print("🚀 Starting Ibani Translator API...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Try different possible paths for the dictionary
    possible_paths = [
        "ibani_dict.json",
        "./ibani_dict.json",
        "/tmp/ibani_dict.json",
        os.path.join(os.path.dirname(__file__), "ibani_dict.json")
    ]
    
    dictionary_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dictionary_path = path
            print(f"✅ Found dictionary at: {path}")
            break
    
    if not dictionary_path:
        print("❌ Dictionary file not found in any expected location")
        print(f"Available files: {os.listdir('.')}")
        return
    
    # Initialize rule-based translator
    try:
        rule_based_translator = IbaniRuleBasedTranslator(dictionary_path)
        print("✅ Rule-based translator initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing rule-based translator: {e}")
        import traceback
        traceback.print_exc()
    
    # Initialize model-based translator
    try:
        # Check if trained model exists
        model_path = "./ibani_model"
        if os.path.exists(model_path):
            print(f"🤖 Loading trained model from {model_path}")
            model_translator = IbaniHuggingFaceTranslator(model_path=model_path)
        else:
            print("🤖 Loading pre-trained model (no fine-tuned model found)")
            model_translator = IbaniHuggingFaceTranslator()
        print("✅ Model-based translator initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing model-based translator: {e}")
        import traceback
        traceback.print_exc()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ibani Translator API",
        "version": "1.0.0",
        "methods": ["rule_based", "model"] if model_translator else ["rule_based"],
        "rule_based_available": rule_based_translator is not None,
        "model_available": model_translator is not None,
        "endpoints": {
            "translate": "/translate (supports method parameter)",
            "batch_translate": "/batch_translate (supports method parameter)",
            "model_translate": "/model/translate",
            "model_batch_translate": "/model/batch_translate",
            "health": "/health",
            "model_health": "/model/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rule_based_available": rule_based_translator is not None,
        "model_available": model_translator is not None,
        "methods": ["rule_based", "model"] if model_translator else ["rule_based"]
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate a single text from English to Ibani using specified method."""
    try:
        if request.method == "model":
            if model_translator is None:
                raise HTTPException(status_code=500, detail="Model-based translator not available")
            
            translated_text = model_translator.translate(request.text)
            confidence = 0.7  # Model-based confidence estimate
            
        else:  # rule_based (default)
            if rule_based_translator is None:
                raise HTTPException(status_code=500, detail="Rule-based translator not available")
            
            translated_text = rule_based_translator.translate_sentence(
                request.text, 
                tense=request.tense
            )
            confidence = 0.8  # Rule-based confidence estimate
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            method=request.method,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.post("/batch_translate", response_model=BatchTranslationResponse)
async def batch_translate_texts(request: BatchTranslationRequest):
    """Translate multiple texts from English to Ibani using specified method."""
    try:
        if request.method == "model":
            if model_translator is None:
                raise HTTPException(status_code=500, detail="Model-based translator not available")
            
            # Use batch translation for efficiency
            translated_texts = model_translator.batch_translate(request.texts)
            translations = []
            
            for i, text in enumerate(request.texts):
                translations.append(TranslationResponse(
                    original_text=text,
                    translated_text=translated_texts[i],
                    method="model",
                    confidence=0.7
                ))
        
        else:  # rule_based (default)
            if rule_based_translator is None:
                raise HTTPException(status_code=500, detail="Rule-based translator not available")
            
            translations = []
            
            for text in request.texts:
                translated_text = rule_based_translator.translate_sentence(
                    text, 
                    tense=request.tense
                )
                confidence = 0.8
                
                translations.append(TranslationResponse(
                    original_text=text,
                    translated_text=translated_text,
                    method="rule_based",
                    confidence=confidence
                ))
        
        return BatchTranslationResponse(translations=translations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")


# Model-based specific endpoints
@app.post("/model/translate", response_model=TranslationResponse)
async def model_translate_text(request: TranslationRequest):
    """Translate using model-based approach only."""
    request.method = "model"
    return await translate_text(request)


@app.post("/model/batch_translate", response_model=BatchTranslationResponse)
async def model_batch_translate_texts(request: BatchTranslationRequest):
    """Batch translate using model-based approach only."""
    request.method = "model"
    return await batch_translate_texts(request)


@app.get("/model/health")
async def model_health_check():
    """Health check for model-based translator."""
    return {
        "status": "healthy" if model_translator is not None else "unavailable",
        "model_available": model_translator is not None,
        "method": "model"
    }


@app.get("/dictionary")
async def get_dictionary():
    """Get the current Ibani dictionary."""
    try:
        # Try different possible paths for the dictionary
        possible_paths = [
            "ibani_dict.json",
            "./ibani_dict.json",
            os.path.join(os.path.dirname(__file__), "ibani_dict.json")
        ]
        
        dictionary_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dictionary_path = path
                break
        
        if not dictionary_path:
            raise HTTPException(status_code=404, detail="Dictionary file not found")
        
        with open(dictionary_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
        return {
            "dictionary": dictionary,
            "total_entries": len(dictionary),
            "format": "comprehensive"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dictionary: {str(e)}")


@app.post("/dictionary")
async def update_dictionary(entry: dict):
    """Add or update a dictionary entry in the comprehensive format."""
    try:
        # Try different possible paths for the dictionary
        possible_paths = [
            "ibani_dict.json",
            "./ibani_dict.json",
            os.path.join(os.path.dirname(__file__), "ibani_dict.json")
        ]
        
        dictionary_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dictionary_path = path
                break
        
        if not dictionary_path:
            raise HTTPException(status_code=404, detail="Dictionary file not found")
        
        with open(dictionary_path, "r", encoding="utf-8") as f:
            dictionary = json.load(f)
        
        # Validate entry format
        required_fields = ["Ibani_word", "Pos", "word"]
        if not all(field in entry for field in required_fields):
            raise HTTPException(
                status_code=400, 
                detail="Entry must contain: Ibani_word, Pos, word"
            )
        
        # Add new entry
        dictionary.append(entry)
        
        # Save updated dictionary
        with open(dictionary_path, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
        
        return {
            "message": "Dictionary updated successfully", 
            "entry": entry,
            "total_entries": len(dictionary)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating dictionary: {str(e)}")


if __name__ == "__main__":
    print("🌐 Starting Ibani Translator API Server...")
    print("📖 API Documentation available at: http://localhost:8080/docs")
    print("🔗 Interactive API at: http://localhost:8080")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )