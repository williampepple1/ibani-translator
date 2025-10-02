"""
FastAPI server for English to Ibani translation.
Provides REST API endpoints for both rule-based and ML-based translation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
from rule_based_translator import IbaniRuleBasedTranslator
from huggingface_translator import IbaniHuggingFaceTranslator
import os


# Initialize FastAPI app
app = FastAPI(
    title="Ibani Translator API",
    description="English to Ibani translation service with rule-based and ML-based approaches",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize translators
rule_based_translator = None
ml_translator = None


class TranslationRequest(BaseModel):
    text: str
    method: str = "rule_based"  # "rule_based" or "ml"
    tense: str = "present"


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    method: str
    confidence: Optional[float] = None


class BatchTranslationRequest(BaseModel):
    texts: List[str]
    method: str = "rule_based"
    tense: str = "present"


class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize translators on startup."""
    global rule_based_translator, ml_translator
    
    print("🚀 Starting Ibani Translator API...")
    
    # Initialize rule-based translator
    try:
        rule_based_translator = IbaniRuleBasedTranslator()
        print("✅ Rule-based translator initialized")
    except Exception as e:
        print(f"❌ Error initializing rule-based translator: {e}")
    
    # Initialize ML translator (if model exists)
    try:
        if os.path.exists("./ibani_model"):
            ml_translator = IbaniHuggingFaceTranslator(model_path="./ibani_model")
            print("✅ ML translator initialized with fine-tuned model")
        else:
            ml_translator = IbaniHuggingFaceTranslator()
            print("✅ ML translator initialized with pre-trained model")
    except Exception as e:
        print(f"❌ Error initializing ML translator: {e}")
        ml_translator = None
    
    yield


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ibani Translator API",
        "version": "1.0.0",
        "endpoints": {
            "translate": "/translate",
            "batch_translate": "/batch_translate",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rule_based_available": rule_based_translator is not None,
        "ml_available": ml_translator is not None
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate a single text from English to Ibani."""
    try:
        if request.method == "rule_based":
            if rule_based_translator is None:
                raise HTTPException(status_code=500, detail="Rule-based translator not available")
            
            translated_text = rule_based_translator.translate_sentence(
                request.text, 
                tense=request.tense
            )
            confidence = 0.8  # Rule-based confidence estimate
            
        elif request.method == "ml":
            if ml_translator is None:
                raise HTTPException(status_code=500, detail="ML translator not available")
            
            translated_text = ml_translator.translate(request.text)
            confidence = 0.9  # ML-based confidence estimate
            
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'rule_based' or 'ml'")
        
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
    """Translate multiple texts from English to Ibani."""
    try:
        translations = []
        
        for text in request.texts:
            if request.method == "rule_based":
                if rule_based_translator is None:
                    raise HTTPException(status_code=500, detail="Rule-based translator not available")
                
                translated_text = rule_based_translator.translate_sentence(
                    text, 
                    tense=request.tense
                )
                confidence = 0.8
                
            elif request.method == "ml":
                if ml_translator is None:
                    raise HTTPException(status_code=500, detail="ML translator not available")
                
                translated_text = ml_translator.translate(text)
                confidence = 0.9
                
            else:
                raise HTTPException(status_code=400, detail="Invalid method. Use 'rule_based' or 'ml'")
            
            translations.append(TranslationResponse(
                original_text=text,
                translated_text=translated_text,
                method=request.method,
                confidence=confidence
            ))
        
        return BatchTranslationResponse(translations=translations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch translation error: {str(e)}")


@app.get("/dictionary")
async def get_dictionary():
    """Get the current Ibani dictionary."""
    try:
        with open("ibani_dict.json", "r", encoding="utf-8") as f:
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
        with open("ibani_dict.json", "r", encoding="utf-8") as f:
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
        with open("ibani_dict.json", "w", encoding="utf-8") as f:
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
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
