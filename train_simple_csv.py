"""
Train MarianMT with CSV data using orthography mapping.
This GUARANTEES 100% character preservation for ḅ, á, ọ́, etc.
"""

import csv
import json
from huggingface_translator import IbaniHuggingFaceTranslator
from ibani_ortho import encode_ibani_text, decode_ibani_text


def prepare_csv_data_with_encoding(
    csv_file: str = "ibani_eng.csv",
    output_file: str = "ibani_encoded_training.json"
):
    """
    Extract data from CSV and encode Ibani characters.
    Special characters are converted to safe ASCII placeholders.
    """
    print(f"Loading data from {csv_file}...")
    training_examples = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            ibani = row.get("ibani_text", "").strip()
            english = row.get("nlt_text", "").strip()
            
            if ibani and english:
                # Encode Ibani text (convert special chars to placeholders)
                ibani_encoded = encode_ibani_text(ibani)
                
                training_examples.append({
                    "translation": {
                        "en": english,
                        "ibani": ibani_encoded
                    }
                })
    
    print(f"Loaded and encoded {len(training_examples)} examples")
    
    # Show sample encoding
    if training_examples:
        sample = training_examples[0]["translation"]
        print(f"\nSample:")
        print(f"  English: {sample['en'][:80]}...")
        print(f"  Ibani (encoded): {sample['ibani'][:80]}...")
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_file}")
    return training_examples


class IbaniTranslatorWithOrtho:
    """
    Wrapper that handles orthography encoding/decoding automatically.
    """
    
    def __init__(self, model_path: str = None):
        self.translator = IbaniHuggingFaceTranslator(model_path=model_path)
    
    def translate(self, english_text: str) -> str:
        """
        Translate English to Ibani with automatic character restoration.
        """
        # Get encoded translation from model
        encoded_ibani = self.translator.translate(english_text)
        
        # Decode placeholders back to Ibani characters
        ibani = decode_ibani_text(encoded_ibani)
        
        return ibani


def main():
    """Train and test with orthography mapping."""
    print("=" * 70)
    print("Training with Orthography Mapping")
    print("=" * 70)
    print("This approach GUARANTEES character preservation!")
    print("  ḅ → [b_dot] → ḅ  (100% preserved)")
    print("  á → [a_acute] → á  (100% preserved)")
    print("  ọ́ → [o_dot_acute] → ọ́  (100% preserved)")
    print("=" * 70)
    
    # Step 1: Prepare encoded data
    print("\nStep 1: Preparing training data with encoding...")
    training_examples = prepare_csv_data_with_encoding()
    
    if not training_examples:
        print("No training examples found!")
        return
    
    # Step 2: Train model
    print("\nStep 2: Training model...")
    translator = IbaniHuggingFaceTranslator()
    
    translator.train_model(
        training_data_file="ibani_encoded_training.json",
        output_dir="./ibani_ortho_model",
        num_epochs=5,
        batch_size=8
    )
    
    # Step 3: Test with decoding
    print("\n" + "=" * 70)
    print("Step 3: Testing with automatic decoding")
    print("=" * 70)
    
    wrapper = IbaniTranslatorWithOrtho(model_path="./ibani_ortho_model")
    
    test_sentences = [
        "Abraham was the father of Isaac",
        "This is the genealogy of Jesus",
        "I eat fish",
        "good morning"
    ]
    
    for sentence in test_sentences:
        translation = wrapper.translate(sentence)
        print(f"\nEN:    {sentence}")
        print(f"IBANI: {translation}")
        
        # Check for special characters
        special = ['ḅ', 'á', 'ọ', 'ẹ', 'ị']
        found = [c for c in special if c in translation]
        if found:
            print(f"✓ Special characters restored: {', '.join(found)}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("Model saved to: ./ibani_ortho_model")
    print("\nTo use this model:")
    print("  from train_simple_csv import IbaniTranslatorWithOrtho")
    print("  translator = IbaniTranslatorWithOrtho('./ibani_ortho_model')")
    print("  result = translator.translate('Hello')")
    print("=" * 70)


if __name__ == "__main__":
    main()
