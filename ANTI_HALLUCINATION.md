# Anti-Hallucination Feature

## Overview

The Ibani translator now includes an **anti-hallucination mechanism** that prevents the model from generating translations for words or phrases that don't exist in the training data. Instead of hallucinating a translation, the system returns the original input text.

## How It Works

### 1. Training Data Lookup
When the translator is initialized, it loads all translation pairs from `ibani_eng_training_data.json` into a lookup dictionary:
- English text (normalized and lowercased) → Ibani translation
- Approximately 32,650 translation pairs loaded

### 2. Translation Validation
When you request a translation:
1. **Exact Match Check**: First checks if the input exactly matches any English text in the training data
2. **Fuzzy Match Check**: If no exact match, uses similarity scoring (threshold: 85%) to find close matches
3. **Fallback**: If no match is found, returns the original input text instead of hallucinating

### 3. Similarity Threshold
The fuzzy matching uses a similarity score of **0.85 (85%)** to account for:
- Minor spelling variations
- Punctuation differences
- Case differences (already normalized)

## Usage

### Basic Usage (Validation Enabled by Default)

```python
from huggingface_translator import IbaniHuggingFaceTranslator

# Initialize with training data
translator = IbaniHuggingFaceTranslator(
    model_path="./ibani_model",
    hf_repo="williampepple1/ibani-translator",
    training_data_file="ibani_eng_training_data.json"
)

# Translate - validation is enabled by default
translation = translator.translate("I eat fish")
# Returns: "ịrị finji fíị" (found in training data)

# Try translating something not in training data
translation = translator.translate("I love programming")
# Returns: "I love programming" (original text, not in training data)
```

### Disable Validation (Use Neural Model)

If you want to use the neural model without validation (may hallucinate):

```python
# Disable validation to use pure neural translation
translation = translator.translate("I love programming", use_validation=False)
# Returns: Neural model output (may hallucinate)
```

## API Server

The API server automatically enables anti-hallucination validation:

```bash
# Start the API server
python api_server.py
```

```bash
# Test with curl
curl -X POST "http://localhost:8080/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "I eat fish"}'

# Response for text in training data:
{
  "source": "I eat fish",
  "translation": "ịrị finji fíị",
  "model": "ibani-translator"
}

# Response for text NOT in training data:
{
  "source": "I love programming",
  "translation": "I love programming",
  "model": "ibani-translator"
}
```

## Configuration

### Environment Variables

You can configure the training data file location:

```bash
export TRAINING_DATA_FILE="./ibani_eng_training_data.json"
```

### Adjusting Similarity Threshold

To change the fuzzy matching threshold, modify the `_find_best_match` method:

```python
def _find_best_match(self, text: str, threshold: float = 0.85) -> Optional[str]:
    # Change threshold value (0.0 to 1.0)
    # Higher = stricter matching
    # Lower = more lenient matching
```

## Benefits

1. **No Hallucinations**: Users only get translations that exist in the training material
2. **Transparency**: Users know when a translation doesn't exist (they get their input back)
3. **Data Quality**: Ensures all translations are from verified training data
4. **User Trust**: Users can trust that translations are accurate, not invented

## Testing

Run the test script to see the feature in action:

```bash
python test_anti_hallucination.py
```

This will test various inputs:
- ✓ Phrases in training data (will translate)
- ✓ Phrases NOT in training data (will return original)
- ✓ Gibberish text (will return original)

## Performance Considerations

### Memory
- The lookup dictionary loads ~32,650 translation pairs into memory
- Approximately 3-5 MB of additional memory usage
- Negligible impact for most deployments

### Speed
- **Exact match**: O(1) - instant lookup
- **Fuzzy match**: O(n) where n = number of training pairs
- For 32,650 pairs, fuzzy matching takes ~0.1-0.5 seconds
- Much faster than neural model inference

### Optimization Tips

If you have a very large training dataset and need faster fuzzy matching:

1. **Disable fuzzy matching** - only use exact matches:
   ```python
   def _find_best_match(self, text: str, threshold: float = 1.0):
       # threshold=1.0 means only exact matches
   ```

2. **Use indexing** - implement a more sophisticated indexing system (e.g., n-grams, BK-trees)

3. **Cache results** - add a cache for frequently requested translations

## Limitations

1. **Exact phrases only**: Won't handle compositional translations (combining multiple training examples)
2. **No context**: Doesn't consider context when matching
3. **Case sensitive after normalization**: "Hello" and "HELLO" are treated the same, but "hello world" and "Hello, world!" may not match exactly

## Future Enhancements

Potential improvements:
- [ ] Add word-level matching for single words
- [ ] Implement n-gram based partial matching
- [ ] Add confidence scores to API responses
- [ ] Support for phrase segmentation and composition
- [ ] Caching layer for frequently requested translations
