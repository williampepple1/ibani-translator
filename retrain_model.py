"""
Retrain the Hugging Face model using ibani_dict.json data.
This script generates training examples from the dictionary and retrains the model.
"""

import json
import os
import random
from typing import List, Dict, Tuple
from huggingface_translator import IbaniHuggingFaceTranslator

def generate_training_data_from_dict(dict_file: str = "ibani_dict.json", output_file: str = "retrained_training_data.json"):
    """
    Generate training data from the Ibani dictionary.
    Creates sentence pairs using dictionary entries.
    """
    print("📚 Loading Ibani dictionary...")
    
    with open(dict_file, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    print(f"✅ Loaded {len(dictionary)} dictionary entries")
    
    # Create training examples
    training_examples = []
    
    # Simple word-to-word translations
    for entry in dictionary:
        english_word = entry["word"].lower().strip()
        ibani_word = entry["Ibani_word"]
        
        # Skip if either word is empty or too short
        if len(english_word) < 2 or len(ibani_word) < 1:
            continue
            
        # Create simple sentence pairs in the correct format
        training_examples.append({
            "translation": {
                "en": english_word,
                "ibani": ibani_word
            }
        })
    
    # Generate more complex sentence patterns
    print("🔄 Generating sentence patterns...")
    
    # Common sentence patterns with Ibani grammar
    patterns = [
        ("I {verb}", "a {verb}"),
        ("You {verb}", "wá {verb}"),
        ("He {verb}", "á {verb}"),
        ("She {verb}", "á {verb}"),
        ("We {verb}", "má {verb}"),
        ("They {verb}", "má {verb}"),
        ("The {noun}", "á {noun}"),
        ("A {noun}", "á {noun}"),
        ("This {noun}", "á {noun}"),
        ("That {noun}", "á {noun}"),
    ]
    
    # Get common verbs and nouns from dictionary
    verbs = [entry for entry in dictionary if "v." in entry["Pos"] or "verb" in entry["Pos"].lower()]
    nouns = [entry for entry in dictionary if "n." in entry["Pos"] or "noun" in entry["Pos"].lower()]
    
    # Generate pattern-based examples
    for pattern_en, pattern_ibani in patterns:
        if "{verb}" in pattern_en and verbs:
            for verb_entry in random.sample(verbs, min(10, len(verbs))):
                en_sentence = pattern_en.format(verb=verb_entry["word"].lower())
                ibani_sentence = pattern_ibani.format(verb=verb_entry["Ibani_word"])
                
                training_examples.append({
                    "translation": {
                        "en": en_sentence,
                        "ibani": ibani_sentence
                    }
                })
        
        if "{noun}" in pattern_en and nouns:
            for noun_entry in random.sample(nouns, min(10, len(nouns))):
                en_sentence = pattern_en.format(noun=noun_entry["word"].lower())
                ibani_sentence = pattern_ibani.format(noun=noun_entry["Ibani_word"])
                
                training_examples.append({
                    "translation": {
                        "en": en_sentence,
                        "ibani": ibani_sentence
                    }
                })
    
    # Add some common phrases
    common_phrases = [
        ("hello", "sannu"),
        ("good morning", "barka da safe"),
        ("good evening", "barka da yamma"),
        ("thank you", "na gode"),
        ("please", "don Allah"),
        ("yes", "i"),
        ("no", "a'a"),
        ("how are you", "yaya kake"),
        ("I am fine", "lafiya lau"),
        ("what is your name", "menene sunanka"),
    ]
    
    for en_phrase, ibani_phrase in common_phrases:
        training_examples.append({
            "translation": {
                "en": en_phrase,
                "ibani": ibani_phrase
            }
        })
    
    # Remove duplicates
    seen = set()
    unique_examples = []
    for example in training_examples:
        key = (example["translation"]["en"], example["translation"]["ibani"])
        if key not in seen:
            seen.add(key)
            unique_examples.append(example)
    
    print(f"✅ Generated {len(unique_examples)} unique training examples")
    
    # Save training data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_examples, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved training data to {output_file}")
    return unique_examples

def retrain_model(training_data_file: str = "retrained_training_data.json", 
                  output_dir: str = "./ibani_model_retrained",
                  num_epochs: int = 5,
                  batch_size: int = 4):
    """
    Retrain the model with the new training data.
    """
    print("🤖 Starting model retraining...")
    
    # Initialize translator
    translator = IbaniHuggingFaceTranslator()
    
    # Train the model
    translator.train_model(
        training_data_file=training_data_file,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    print(f"✅ Model retraining completed! Model saved to {output_dir}")
    return output_dir

def test_retrained_model(model_path: str = "./ibani_model_retrained"):
    """
    Test the retrained model with some sample translations.
    """
    print("🧪 Testing retrained model...")
    
    try:
        # Load the retrained model
        translator = IbaniHuggingFaceTranslator(model_path=model_path)
        
        # Test sentences
        test_sentences = [
            "hello",
            "I eat fish",
            "good morning",
            "thank you",
            "how are you",
            "I am fine",
            "water",
            "food",
            "house",
            "man"
        ]
        
        print("\n📝 Translation Results:")
        print("=" * 50)
        
        for sentence in test_sentences:
            try:
                translation = translator.translate(sentence)
                print(f"EN: {sentence}")
                print(f"IBANI: {translation}")
                print("-" * 30)
            except Exception as e:
                print(f"❌ Error translating '{sentence}': {e}")
        
        print("✅ Model testing completed!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def main():
    """Main function to retrain the model."""
    print("🚀 Ibani Model Retraining Pipeline")
    print("=" * 50)
    
    # Step 1: Generate training data from dictionary
    print("\n📚 Step 1: Generating training data from dictionary...")
    training_examples = generate_training_data_from_dict()
    
    # Step 2: Retrain the model
    print("\n🤖 Step 2: Retraining the model...")
    model_path = retrain_model(
        training_data_file="retrained_training_data.json",
        output_dir="./ibani_model_retrained",
        num_epochs=5,
        batch_size=4
    )
    
    # Step 3: Test the retrained model
    print("\n🧪 Step 3: Testing the retrained model...")
    test_retrained_model(model_path)
    
    print("\n🎉 Retraining pipeline completed!")
    print(f"📁 New model saved to: {model_path}")
    print("🔄 Update your API to use the new model by changing the model_path in main.py")

if __name__ == "__main__":
    main()
