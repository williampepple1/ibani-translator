"""
FastAPI server for Ibani translation API endpoint.
"""

import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from huggingface_translator import IbaniHuggingFaceTranslator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Ibani Translation API",
    description="English to Ibani translation API using trained MarianMT model",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    local_model_path = os.getenv("LOCAL_MODEL_PATH", "./ibani_model")
    
    print(f"Local model path: {local_model_path}")
    print(f"HuggingFace repo: {hf_repo}")
    
    translator = IbaniHuggingFaceTranslator(
        model_path=local_model_path,
        hf_repo=hf_repo
    )
    print("✓ Model loaded successfully!")


@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    """Root endpoint with API information."""
    return {
        "message": "Ibani Translation API",
        "version": "1.0.0",
        "endpoints": {
            "/translate": "POST - Translate single text (20 requests/minute)",
            "/batch-translate": "POST - Translate multiple texts (5 requests/minute)",
            "/health": "GET - Check API health (30 requests/minute)"
        },
        "rate_limits": {
            "translate": "20 per minute",
            "batch_translate": "5 per minute",
            "other_endpoints": "30 per minute"
        }
    }


@app.get("/health")
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": "./ibani_model"
    }


@app.post("/translate", response_model=TranslationResponse)
@limiter.limit("20/minute")
async def translate(http_request: Request, request: TranslationRequest):
    """
    Translate English text to Ibani.
    
    Rate limit: 20 requests per minute per IP address.
    
    Args:
        http_request: FastAPI Request object (for rate limiting)
        request: TranslationRequest with text to translate
        
    Returns:
        TranslationResponse with original text and translation
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        translation = translator.translate(request.text)
        return TranslationResponse(
            source=request.text,
            translation=translation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.post("/batch-translate", response_model=BatchTranslationResponse)
@limiter.limit("5/minute")
async def batch_translate(http_request: Request, request: BatchTranslationRequest):
    """
    Translate multiple English texts to Ibani.
    
    Rate limit: 5 requests per minute per IP address.
    Maximum 50 texts per request.
    
    Args:
        http_request: FastAPI Request object (for rate limiting)
        request: BatchTranslationRequest with list of texts
        
    Returns:
        BatchTranslationResponse with all translations
    """
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    # Limit batch size to prevent abuse
    if len(request.texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 texts per batch request"
        )
    
    try:
        translations = []
        for text in request.texts:
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

