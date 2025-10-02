"""
FastAPI server for English to Ibani translation.
Rule-based translation service optimized for Vercel deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
from rule_based_translator import IbaniRuleBasedTranslator
import os


# Initialize FastAPI app
app = FastAPI(
    title="Ibani Translator API",
    description="English to Ibani translation service using rule-based approach",
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

# Initialize translator
rule_based_translator = None


class TranslationRequest(BaseModel):
    text: str
    tense: str = "present"


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    method: str = "rule_based"
    confidence: Optional[float] = None


class BatchTranslationRequest(BaseModel):
    texts: List[str]
    tense: str = "present"


class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]


@app.on_event("startup")
async def startup_event():
    """Initialize translator on startup."""
    global rule_based_translator
    
    print("🚀 Starting Ibani Translator API...")
    
    # Initialize rule-based translator
    try:
        rule_based_translator = IbaniRuleBasedTranslator()
        print("✅ Rule-based translator initialized")
    except Exception as e:
        print(f"❌ Error initializing rule-based translator: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ibani Translator API",
        "version": "1.0.0",
        "method": "rule_based",
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
        "method": "rule_based"
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translate a single text from English to Ibani."""
    try:
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
            method="rule_based",
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.post("/batch_translate", response_model=BatchTranslationResponse)
async def batch_translate_texts(request: BatchTranslationRequest):
    """Translate multiple texts from English to Ibani."""
    try:
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
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )