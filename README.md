# Ibani Translator 🌍

A neural machine translation system for English to Ibani language using Hugging Face transformers.

## Features

- **Neural Translation**: Fine-tuned MarianMT model for English to Ibani translation
- **REST API**: FastAPI service with interactive documentation
- **Training Pipeline**: Train custom models on your own Ibani data
- **Batch Translation**: Translate multiple texts efficiently
- **Model Hosting**: Ready for deployment to Hugging Face Hub

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start API Server

```bash
python api_server.py
```

The server will start at `http://localhost:8080`

Visit `http://localhost:8080/docs` for interactive API documentation.

### 3. Use the API Client

```bash
python api_client.py
```

## Project Structure

```
ibani-translator/
├── api_server.py               # FastAPI server
├── api_client.py               # API client with examples
├── API_USAGE.md               # Comprehensive API documentation
├── huggingface_translator.py  # Neural translation core
├── rule_based_translator.py   # Grammar rules and fallback
├── train_from_ibani_eng.py    # Model training script
├── ibani_dict.json            # Ibani-English dictionary
├── ibani_eng.json             # Training data source
├── ibani_eng.csv              # Training data (CSV format)
├── ibani_eng_training_data.json # Formatted training data
├── ibani_model/               # Trained model files
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### API Translation

#### Single Translation
```bash
curl -X POST "http://localhost:8080/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "I eat fish"}'
```

**Response:**
```json
{
  "source": "I eat fish",
  "translation": "ịrị olokpó fíị",
  "model": "ibani-translator"
}
```

#### Batch Translation
```bash
curl -X POST "http://localhost:8080/batch-translate" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Good morning", "Thank you", "I love you"]}'
```

### Python Usage

```python
from huggingface_translator import IbaniHuggingFaceTranslator

# Load trained model
translator = IbaniHuggingFaceTranslator(model_path="./ibani_model")

# Translate
result = translator.translate("I eat fish")
print(result)
```

### API Client Usage

```python
import requests

response = requests.post(
    "http://localhost:8080/translate",
    json={"text": "I eat fish"}
)

result = response.json()
print(f"Translation: {result['translation']}")
```

## Training Custom Model

### 1. Prepare Your Data

Your data should be in JSON format with English and Ibani parallel texts:

```json
[
  {
    "english_text": "I eat fish",
    "ibani_text": "ịrị olokpó fíị"
  },
  {
    "english_text": "Good morning",
    "ibani_text": "ụ̀bọ́sị́ ọ́má"
  }
]
```

### 2. Train the Model

```bash
python train_from_ibani_eng.py
```

This will:
1. Extract training data from `ibani_eng.json`
2. Train the MarianMT model
3. Save the trained model to `./ibani_model`
4. Test the model with sample translations

### 3. Customize Training

Edit `train_from_ibani_eng.py` to adjust:
- `num_epochs`: Number of training epochs (default: 5)
- `batch_size`: Training batch size (default: 4)
- `learning_rate`: Learning rate (default: 5e-5)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and model status |
| `/translate` | POST | Translate single text |
| `/batch-translate` | POST | Translate multiple texts |
| `/docs` | GET | Interactive API documentation |

For detailed API documentation, see [API_USAGE.md](API_USAGE.md)

## Model Information

- **Base Model**: Helsinki-NLP/opus-mt-en-mul
- **Task**: English → Ibani Translation
- **Framework**: Hugging Face Transformers
- **Model Size**: ~294 MB
- **Training Data**: Parallel English-Ibani sentence pairs

## Deployment

### Deploy to Hugging Face Hub

```python
from huggingface_translator import IbaniHuggingFaceTranslator

translator = IbaniHuggingFaceTranslator(model_path="./ibani_model")
translator.push_to_hub("your-username/ibani-translator")
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "api_server.py"]
```

Build and run:
```bash
docker build -t ibani-translator .
docker run -p 8080:8080 ibani-translator
```

## Dependencies

- `transformers>=4.30.0` - Hugging Face transformers
- `torch>=2.0.0` - PyTorch for ML models
- `fastapi>=0.100.0` - Web API framework
- `uvicorn>=0.22.0` - ASGI server
- `datasets>=2.12.0` - Dataset handling
- `sentencepiece>=0.1.99` - Tokenization

## Performance

- **Translation Speed**: ~100-200ms per sentence (CPU)
- **Batch Processing**: More efficient for multiple texts
- **GPU Support**: Automatic CUDA detection for faster inference
- **Model Cache**: First load is slower, subsequent loads are instant

## Troubleshooting

### Common Issues

**Problem**: Cannot connect to API
- **Solution**: Ensure server is running with `python api_server.py`

**Problem**: Model not found
- **Solution**: Train the model first with `python train_from_ibani_eng.py`

**Problem**: Out of memory during training
- **Solution**: Reduce `batch_size` in training parameters

**Problem**: Slow translations
- **Solution**: Use GPU if available, or reduce `num_beams` parameter

**Problem**: Port 8080 already in use
- **Solution**: Change port in `api_server.py`: `uvicorn.run(app, port=8081)`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Feel free to use and modify for your Ibani language needs.

## Support

- Interactive API Docs: http://localhost:8080/docs
- API Usage Guide: [API_USAGE.md](API_USAGE.md)
- Check troubleshooting section above

---

**Happy Translating! 🌍➡️🇳🇬**
