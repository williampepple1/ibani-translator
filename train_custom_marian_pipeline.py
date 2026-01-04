"""
Complete pipeline: Train custom tokenizer + Train MarianMT with it.
This is the "Custom Tokenizer + MarianMT" approach.
"""

import csv
import json
import os
from train_custom_tokenizer import extract_ibani_text_from_csv, train_sentencepiece_tokenizer, test_tokenizer
from custom_marian_translator import CustomMarianTranslator


def prepare_training_data_from_csv(
    input_file: str = "ibani_eng.csv",
    output_file: str = "ibani_eng_custom_marian_data.json"
):
    """Extract training data from CSV."""
    print(f"Loading data from {input_file}...")
    training_examples = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            ibani = row.get("ibani_text", "").strip()
            english = row.get("nlt_text", "").strip()
            
            if ibani and english:
                training_examples.append({
                    "translation": {
                        "en": english,
                        "ibani": ibani
                    }
                })
    
    print(f"Loaded {len(training_examples)} examples")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_file}")
    return training_examples


def main():
    """Complete training pipeline."""
    print("="*70)
    print("Custom Tokenizer + MarianMT Training Pipeline")
    print("="*70)
    print("This approach:")
    print("  ✓ Trains custom tokenizer with ALL Ibani characters")
    print("  ✓ Uses MarianMT for fast training/inference")
    print("  ✓ Perfect character preservation (ḅ, á, ọ́, etc.)")
    print("  ✓ Faster than ByT5")
    print("="*70)
    
    # Step 1: Train custom tokenizer (if not exists)
    tokenizer_file = "ibani_tokenizer.model"
    if not os.path.exists(tokenizer_file):
        print("\nStep 1: Training custom SentencePiece tokenizer...")
        print("-"*70)
        
        # Extract text
        all_text, ibani_texts, english_texts = extract_ibani_text_from_csv()
        
        # Train tokenizer
        train_sentencepiece_tokenizer(
            text_data=all_text,
            model_prefix="ibani_tokenizer",
            vocab_size=8000,
            character_coverage=1.0  # 100% character coverage
        )
        
        # Test tokenizer
        test_sentences = [
            "Mịị anịị diri bie anị fịnị ḅara Jizọs tádọ́apụ",
            "Ebraham anịị Aizik daa",
            "Abraham was the father of Isaac"
        ]
        test_tokenizer(tokenizer_file, test_sentences)
    else:
        print(f"\n✓ Custom tokenizer already exists: {tokenizer_file}")
    
    # Step 2: Prepare training data
    print("\nStep 2: Preparing training data...")
    print("-"*70)
    training_data_file = "ibani_eng_custom_marian_data.json"
    if not os.path.exists(training_data_file):
        prepare_training_data_from_csv(output_file=training_data_file)
    else:
        print(f"✓ Training data already exists: {training_data_file}")
    
    # Step 3: Train MarianMT with custom tokenizer
    print("\nStep 3: Training MarianMT with custom tokenizer...")
    print("-"*70)
    
    translator = CustomMarianTranslator(tokenizer_model=tokenizer_file)
    
    translator.train_model(
        training_data_file=training_data_file,
        output_dir="./ibani_custom_marian_model",
        num_epochs=5,
        batch_size=8,
        learning_rate=5e-5
    )
    
    # Step 4: Test the model
    print("\nStep 4: Testing trained model...")
    print("-"*70)
    
    # Load trained model
    trained_translator = CustomMarianTranslator(
        tokenizer_model=tokenizer_file,
        model_path="./ibani_custom_marian_model"
    )
    
    test_sentences = [
        "This is the genealogy of Jesus the Messiah the son of David",
        "Abraham was the father of Isaac",
        "I eat fish",
        "good morning",
        "thank you"
    ]
    
    print("\nTranslation Results:")
    print("="*70)
    for sentence in test_sentences:
        translation = trained_translator.translate(sentence)
        print(f"EN:    {sentence}")
        print(f"IBANI: {translation}")
        
        # Check for special characters
        special_chars = ['ḅ', 'á', 'ọ', 'ẹ', 'í', 'ú', 'ó']
        found = [c for c in special_chars if c in translation]
        if found:
            print(f"✓ Special characters: {', '.join(found)}")
        print("-"*70)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print("\nFiles created:")
    print("  - ibani_tokenizer.model (Custom tokenizer)")
    print("  - ibani_tokenizer.vocab (Vocabulary)")
    print("  - ibani_custom_marian_model/ (Trained model)")
    print("\nAdvantages of this approach:")
    print("  ✓ Faster than ByT5")
    print("  ✓ Perfect character preservation")
    print("  ✓ Smaller model size")
    print("  ✓ You control the vocabulary")
    print("="*70)


if __name__ == "__main__":
    # Check dependencies
    try:
        import sentencepiece
        main()
    except ImportError:
        print("ERROR: sentencepiece not installed")
        print("Install it with: pip install sentencepiece")
