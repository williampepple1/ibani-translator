"""
Train the model using ibani_eng.csv file.
Extracts only ibani_text and nlt_text fields for training.
"""

import csv
import json
import traceback
from typing import List, Dict
from huggingface_translator import IbaniHuggingFaceTranslator


def prepare_training_data_from_csv(
    input_file: str = "ibani_eng.csv",
    output_file: str = "ibani_eng_csv_training_data.json"
) -> List[Dict[str, Dict[str, str]]]:
    """
    Extract ibani_text and nlt_text from CSV file for training.
    """
    print(f"Loading data from {input_file}...")
    training_examples = []
    
    # Load CSV data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                # Extract the relevant fields
                ibani = row.get("ibani_text", "").strip()
                english = row.get("nlt_text", "").strip()
                
                # Only add if both fields have content
                if ibani and english:
                    training_examples.append({
                        "translation": {
                            "en": english,
                            "ibani": ibani
                        }
                    })
        
        print(f"Successfully loaded {len(training_examples)} examples from CSV")
        
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        traceback.print_exc()
        raise

    # Add identity mappings for common names and English words (Copy Task)
    # This helps the model realize some words shouldn't be "translated" to garbage
    common_identities = [
        "John", "Mary", "Jesus", "David", "Abraham", "Peter", "Paul", 
        "James", "Joseph", "Simon", "Matthew", "Andrew",
        "Lagos", "Nigeria", "Internet", "Computer"
    ]
    for word in common_identities:
        training_examples.append({
            "translation": {
                "en": word,
                "ibani": word
            }
        })

    print(f"Total training examples (including identity mappings): {len(training_examples)}")
    
    # Save training data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, ensure_ascii=False, indent=2)
    
    print(f"Training data saved to {output_file}")
    return training_examples


def train_model_with_csv_data(
    training_data_file: str = "ibani_eng_csv_training_data.json",
    output_dir: str = "./ibani_csv_model",
    num_epochs: int = 5,
    batch_size: int = 4
) -> str:
    """
    Train the model using the prepared training data from CSV.
    
    Args:
        training_data_file: Path to the training data JSON file
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Path to the saved model directory
    """
    print("Starting model training...")
    
    try:
        # Initialize translator
        translator = IbaniHuggingFaceTranslator()
        
        # Train the model
        translator.train_model(
            training_data_file=training_data_file,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        
        print(f"Model training completed! Model saved to {output_dir}")
        return output_dir
    except (ValueError, RuntimeError, OSError, FileNotFoundError, 
            AttributeError, TypeError) as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()
        raise


def test_trained_model(model_path: str = "./ibani_csv_model") -> None:
    """
    Test the trained model with some sample translations.
    
    Args:
        model_path: Path to the trained model directory
    """
    print("Testing trained model...")
    
    try:
        # Load the trained model
        translator = IbaniHuggingFaceTranslator(model_path=model_path)
        
        # Test sentences - including some with special characters
        test_sentences = [
            "This is the genealogy of Jesus the Messiah the son of David, the son of Abraham:",
            "Abraham was the father of Isaac, Isaac the father of Jacob, Jacob the father of Judah and his brothers,",
            "I eat fish",
            "good morning",
            "thank you",
            "how are you",
            "I am fine",
            "God bless you"
        ]
        
        print("\nTranslation Results:")
        print("=" * 70)
        
        for sentence in test_sentences:
            try:
                translation = translator.translate(sentence)
                print(f"EN: {sentence}")
                print(f"IBANI: {translation}")
                print("-" * 70)
            except (ValueError, RuntimeError, AttributeError, TypeError, 
                    IndexError) as e:
                print(f"Error translating '{sentence}': {e}")
                traceback.print_exc()
        
        print("Model testing completed!")
        
    except (FileNotFoundError, OSError, ValueError, RuntimeError, 
            AttributeError, TypeError) as e:
        print(f"Error testing model: {e}")
        traceback.print_exc()


def main() -> None:
    """Main function to train model from ibani_eng.csv."""
    print("Ibani Model Training from ibani_eng.csv")
    print("=" * 70)
    print("This script will extract ibani_text and nlt_text fields from the CSV")
    print("to train the translation model with proper character support.")
    print("=" * 70)
    
    # Step 1: Prepare training data
    print("\nStep 1: Preparing training data from ibani_eng.csv...")
    try:
        training_examples = prepare_training_data_from_csv()
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"Failed to prepare training data: {e}")
        return
    
    if len(training_examples) == 0:
        print("No training examples found! Exiting.")
        return
    
    # Step 2: Train the model
    print("\nStep 2: Training the model...")
    try:
        model_path = train_model_with_csv_data(
            training_data_file="ibani_eng_csv_training_data.json",
            output_dir="./ibani_csv_model",
            num_epochs=5,
            batch_size=4
        )
    except (ValueError, RuntimeError, OSError, FileNotFoundError, 
            AttributeError, TypeError) as e:
        print(f"Failed to train model: {e}")
        return
    
    # Step 3: Test the trained model
    print("\nStep 3: Testing the trained model...")
    test_trained_model(model_path)
    
    print("\n" + "=" * 70)
    print("Training pipeline completed!")
    print(f"Model saved to: {model_path}")
    print("\nTo use this model in your API, update the model_path in api_server.py")
    print("or copy the model to ./ibani_model directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
