# Ibani Translation API Usage Guide

## Quick Start

### 1. Start the API Server

```bash
python api_server.py
```

The server will start at `http://localhost:8080`

### 2. Use the API

**Option A: Use the Python client**
```bash
python api_client.py
```

**Option B: Use curl commands** (see below)

**Option C: Use any HTTP client** (Postman, Thunder Client, etc.)

---

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and the model is loaded.

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "./ibani_model"
}
```

---

### 2. Single Translation
**POST** `/translate`

Translate a single English text to Ibani.

**Request Body:**
```json
{
  "text": "I eat fish",
  "max_length": 128,
  "num_beams": 4
}
```

**Using curl:**
```bash
curl -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I eat fish\"}"
```

**Using PowerShell:**
```powershell
$body = @{
    text = "I eat fish"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/translate" -Method Post -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "source": "I eat fish",
  "translation": "ịrị olokpó fíị",
  "model": "ibani-translator"
}
```

---

### 3. Batch Translation
**POST** `/batch-translate`

Translate multiple texts at once.

**Request Body:**
```json
{
  "texts": [
    "Good morning",
    "Thank you",
    "I love you"
  ],
  "max_length": 128,
  "num_beams": 4
}
```

**Using curl:**
```bash
curl -X POST http://localhost:8080/batch-translate \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"Good morning\", \"Thank you\", \"I love you\"]}"
```

**Using PowerShell:**
```powershell
$body = @{
    texts = @("Good morning", "Thank you", "I love you")
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/batch-translate" -Method Post -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "translations": [
    {
      "source": "Good morning",
      "translation": "ụ̀tụ̀tụ ọma",
      "model": "ibani-translator"
    },
    {
      "source": "Thank you",
      "translation": "daalụ",
      "model": "ibani-translator"
    },
    {
      "source": "I love you",
      "translation": "ahụrụ m gị n'anya",
      "model": "ibani-translator"
    }
  ],
  "count": 3
}
```

---

## Python Examples

### Example 1: Simple Translation
```python
import requests

response = requests.post(
    "http://localhost:8080/translate",
    json={"text": "I eat fish"}
)

result = response.json()
print(f"Translation: {result['translation']}")
```

### Example 2: Batch Translation
```python
import requests

response = requests.post(
    "http://localhost:8080/batch-translate",
    json={
        "texts": [
            "Good morning",
            "Thank you",
            "How are you"
        ]
    }
)

results = response.json()
for item in results['translations']:
    print(f"{item['source']} → {item['translation']}")
```

### Example 3: Error Handling
```python
import requests

try:
    response = requests.post(
        "http://localhost:8080/translate",
        json={"text": "Hello world"},
        timeout=10
    )
    response.raise_for_status()
    result = response.json()
    print(result['translation'])
except requests.exceptions.ConnectionError:
    print("Error: Cannot connect to API")
except requests.exceptions.Timeout:
    print("Error: Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"Error: {e}")
```

---

## JavaScript/Node.js Example

```javascript
// Using fetch
async function translateText(text) {
    const response = await fetch('http://localhost:8080/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    });
    
    const result = await response.json();
    return result.translation;
}

// Usage
translateText("I eat fish").then(translation => {
    console.log(translation);
});
```

---

## API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8080/docs
- **Alternative Docs**: http://localhost:8080/redoc

These provide a full interactive API documentation where you can test endpoints directly in your browser!

---

## Deployment Options

### Option 1: Local Network Access
Change `host="0.0.0.0"` in `api_server.py` to allow access from other devices on your network.

### Option 2: Deploy to Cloud
- **Heroku**: Use Procfile
- **AWS Lambda**: Use Mangum adapter
- **Google Cloud Run**: Use Docker
- **Azure**: Use Azure Functions

### Option 3: Docker Deployment
Create a `Dockerfile`:
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

---

## Performance Tips

1. **Keep the server running** - Model loading takes time, so keep the API running
2. **Use batch translation** for multiple texts - More efficient than individual requests
3. **Adjust `num_beams`** - Lower values (2-3) are faster, higher (4-5) are more accurate
4. **Add caching** - Cache common translations using Redis or similar

---

## Troubleshooting

**Problem**: Cannot connect to API
- **Solution**: Make sure the server is running with `python api_server.py`

**Problem**: Model not loaded error
- **Solution**: Ensure `./ibani_model` directory exists and contains the trained model

**Problem**: Slow translations
- **Solution**: Use GPU if available, or reduce `num_beams` parameter

**Problem**: Port 8080 already in use
- **Solution**: Change the port in `api_server.py`: `uvicorn.run(app, port=8001)`

