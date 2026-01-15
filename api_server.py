"""
FastAPI server for Ibani translation API endpoint.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from huggingface_translator import IbaniHuggingFaceTranslator
import uvicorn


# Initialize FastAPI app
app = FastAPI(
    title="Ibani Translation API",
    description="English to Ibani translation API using trained MarianMT model",
    version="1.0.0"
)

# Global translator instance
translator = None


class TranslationRequest(BaseModel):
    """Request model for translation."""
    text: str
    max_length: Optional[int] = 128
    num_beams: Optional[int] = 4


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation."""
    texts: List[str]
    max_length: Optional[int] = 128
    num_beams: Optional[int] = 4


class TranslationResponse(BaseModel):
    """Response model for translation."""
    source: str
    translation: str
    model: str = "ibani-translator"


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation."""
    translations: List[TranslationResponse]
    count: int


@app.on_event("startup")
async def load_model():
    """Load the model when the server starts."""
    global translator
    print("Loading Ibani translation model...")
    
    # Get HuggingFace repo from environment variable or use default
    hf_repo = os.getenv("HF_MODEL_REPO", "williampepple1/ibani-translator")
    # Use the trained model directory
    local_model_path = os.getenv("LOCAL_MODEL_PATH", "./ibani_model")
    training_data_file = os.getenv("TRAINING_DATA_FILE", "./ibani_eng_training_data.json")
    
    print(f"Local model path: {local_model_path}")
    print(f"HuggingFace repo: {hf_repo}")
    print(f"Training data file: {training_data_file}")
    
    translator = IbaniHuggingFaceTranslator(
        model_path=local_model_path,
        hf_repo=hf_repo,
        training_data_file=training_data_file
    )
    print("✓ Model loaded successfully!")I wa


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ibani Translation API",
        "version": "1.0.0",
        "endpoints": {
            "/translate": "POST - Translate single text",
            "/batch-translate": "POST - Translate multiple texts",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": "./ibani_model"
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate(payload: TranslationRequest):
    """
    Translate English text to Ibani.
    
    Args:
        payload: TranslationRequest with text to translate
        
    Returns:
        TranslationResponse with original text and translation
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        translation = translator.translate(payload.text)
        return TranslationResponse(
            source=payload.text,
            translation=translation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.post("/batch-translate", response_model=BatchTranslationResponse)
async def batch_translate(payload: BatchTranslationRequest):
    """
    Translate multiple English texts to Ibani.
    
    Maximum 50 texts per request.
    
    Args:
        payload: BatchTranslationRequest with list of texts
        
    Returns:
        BatchTranslationResponse with all translations
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not payload.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    # Limit batch size to prevent abuse
    if len(payload.texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 texts per batch request"
        )
    
    try:
        translations = []
        for text in payload.texts:
            if text.strip():
                translation = translator.translate(text)
                translations.append(TranslationResponse(
                    source=text,
                    translation=translation
                ))
        
        return BatchTranslationResponse(
            translations=translations,
            count=len(translations)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


def main():
    """Run the API server."""
    print("Starting Ibani Translation API Server...")
    print("=" * 60)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )


if __name__ == "__main__":
    main()
