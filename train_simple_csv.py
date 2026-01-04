"""
Simple solution: Train the existing MarianMT model with CSV data.
No custom tokenizer complexity - just use what works!
"""

import csv
import json
from huggingface_translator import IbaniHuggingFaceTranslator


def prepare_csv_data_for_training(
    csv_file: str = "ibani_eng.csv",
    output_file: str = "ibani_csv_simple_training.json"
):
    """Extract data from CSV for training."""
    print(f"Loading data from {csv_file}...")
    training_examples = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
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
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_file}")
    return training_examples


def main():
    """Train model using existing, proven code."""
    print("=" * 70)
    print("Simple CSV Training - Using Proven MarianMT Approach")
    print("=" * 70)
    
    # Step 1: Prepare data
    print("\nStep 1: Preparing training data from CSV...")
    training_examples = prepare_csv_data_for_training()
    
    if len(training_examples) == 0:
        print("No training examples found!")
        return
    
    # Step 2: Train using existing translator
    print("\nStep 2: Training model with existing MarianMT translator...")
    print("This uses the proven approach with character augmentation.")
    
    translator = IbaniHuggingFaceTranslator()
    
    translator.train_model(
        training_data_file="ibani_csv_simple_training.json",
        output_dir="./ibani_simple_model",
        num_epochs=5,
        batch_size=8
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("Model saved to: ./ibani_simple_model")
    print("=" * 70)
    
    # Step 3: Test
    print("\nTesting model...")
    trained_translator = IbaniHuggingFaceTranslator(model_path="./ibani_simple_model")
    
    test_sentences = [
        "Abraham was the father of Isaac",
        "I eat fish",
        "good morning"
    ]
    
    for sentence in test_sentences:
        translation = trained_translator.translate(sentence)
        print(f"EN: {sentence}")
        print(f"IBANI: {translation}\n")


if __name__ == "__main__":
    main()
