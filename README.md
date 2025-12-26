# Ibani Translator 🌍

An English to Ibani translation system using both rule-based and machine learning approaches with Hugging Face transformers.

## Features

- **Rule-based Translation**: Grammar-aware translation with Ibani syntax rules
- **ML-based Translation**: Neural machine translation using Hugging Face MarianMT
- **REST API**: FastAPI service for easy integration
- **Training Pipeline**: Custom model training on Ibani data
- **Dictionary Management**: Extensible Ibani-English dictionary

## Quick Start

### 1. Setup Virtual Environment

```bash
# Windows
python setup.py
# Then activate with:
venv\Scripts\activate

# Linux/Mac
python setup.py
# Then activate with:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Rule-based Translator

```bash
python rule_based_translator.py
```

### 4. Run ML-based Translator

```bash
python huggingface_translator.py
```

### 5. Start API Server

```bash
python api_server.py
```

Visit `http://localhost:8080/docs` for interactive API documentation.

## Project Structure

```
ibani-translator/
├── requirements.txt          # Python dependencies
├── setup.py                 # Environment setup script
├── activate_env.bat         # Windows activation script
├── activate_env.sh          # Linux/Mac activation script
├── ibani_dict.json          # Ibani-English dictionary
├── rule_based_translator.py # Rule-based translation
├── huggingface_translator.py # ML-based translation
├── api_server.py            # FastAPI server
├── training_data.json       # Training dataset (auto-generated)
└── README.md               # This file
```

## Usage Examples

### Rule-based Translation

```python
from rule_based_translator import IbaniRuleBasedTranslator

translator = IbaniRuleBasedTranslator()
result = translator.translate_sentence("I eat fish")
print(result)  # Output: "mi sibi bia"
```

### ML-based Translation

```python
from huggingface_translator import IbaniHuggingFaceTranslator

translator = IbaniHuggingFaceTranslator()
result = translator.translate("I eat fish")
print(result)  # Output: "Mi sibi bia"
```

### API Usage

```bash
# Single translation
curl -X POST "http://localhost:8080/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "I eat fish", "method": "rule_based"}'

# Batch translation
curl -X POST "http://localhost:8080/batch_translate" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["I eat fish", "The woman goes"], "method": "ml"}'
```

## Training Custom Model

### 1. Prepare Training Data

Create `training_data.json` with parallel English-Ibani sentences:

```json
[
  {"translation": {"en": "I eat fish", "ibani": "Mi sibi bia"}},
  {"translation": {"en": "The woman goes", "ibani": "Inyengi zigha"}}
]
```

### 2. Train the Model

```python
from huggingface_translator import IbaniHuggingFaceTranslator

translator = IbaniHuggingFaceTranslator()
translator.train_model(
    training_data_file="training_data.json",
    output_dir="./ibani_model",
    num_epochs=5,
    batch_size=4
)
```

## Grammar Rules

The rule-based translator implements Ibani grammar rules:

- **Word Order**: Subject-Object-Verb (SOV)
- **Tense Markers**: Present, past, future tense handling
- **Negation**: Prefix-based negation
- **Pronouns**: Proper pronoun translation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/translate` | POST | Single text translation |
| `/batch_translate` | POST | Multiple text translation |
| `/dictionary` | GET | View dictionary |
| `/dictionary` | POST | Update dictionary |

## Configuration

### Dictionary Management

Add new words to `ibani_dict.json`:

```json
{
  "hello": "tam",
  "goodbye": "tam",
  "thank you": "tam"
}
```

### Model Configuration

Modify training parameters in `huggingface_translator.py`:

```python
translator.train_model(
    num_epochs=10,        # Training epochs
    batch_size=8,         # Batch size
    learning_rate=1e-5    # Learning rate
)
```

## Dependencies

- `transformers>=4.30.0` - Hugging Face transformers
- `torch>=2.0.0` - PyTorch for ML models
- `fastapi>=0.100.0` - Web API framework
- `datasets>=2.12.0` - Dataset handling
- `sentencepiece>=0.1.99` - Tokenization

## Troubleshooting

### Common Issues

1. **CUDA not available**: The model will use CPU if CUDA is not available
2. **Memory issues**: Reduce batch size in training
3. **Dictionary not found**: Ensure `ibani_dict.json` exists
4. **Model loading errors**: Check if model files exist in the specified path

### Performance Tips

- Use GPU for faster training and inference
- Increase batch size if you have more memory
- Use smaller models for faster inference
- Cache translations for repeated requests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Feel free to use and modify for your Ibani language needs.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the API documentation at `/docs`
- Test with the provided examples

---

**Happy Translating! 🌍➡️🇳🇬**
