"""
Train the model using ibani_eng.json file.
Extracts only ibani_text and english_text fields for training.
"""

import json
import traceback
from typing import List, Dict
from huggingface_translator import IbaniHuggingFaceTranslator


def prepare_training_data_from_ibani_eng(
    input_file: str = "ibani_eng.json",
    output_file: str = "ibani_eng_training_data.json"
) -> List[Dict[str, Dict[str, str]]]:
    """
    Extract ibani_text and english_text from ibani_eng.json and format for training.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the formatted training data
        
    Returns:
        List of training examples in the required format
    """
    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        raise
    
    print(f"Loaded {len(data)} entries from {input_file}")
    
    # Extract only ibani_text and english_text
    training_examples = []
    
    for entry in data:
        english_text = entry.get("english_text", "").strip()
        ibani_text = entry.get("ibani_text", "").strip()
        
        # Skip if either field is empty
        if not english_text or not ibani_text:
            continue
        
        # Format for Hugging Face training (needs "translation" field)
        training_examples.append({
            "translation": {
                "en": english_text,
                "ibani": ibani_text
            }
        })
    
    print(f"Extracted {len(training_examples)} valid training examples")
    
    # Save training data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, ensure_ascii=False, indent=2)
        print(f"Saved training data to {output_file}")
    except IOError as e:
        print(f"Error saving training data to '{output_file}': {e}")
        raise
    
    return training_examples


def train_model_with_ibani_eng_data(
    training_data_file: str = "ibani_eng_training_data.json",
    output_dir: str = "./ibani_model",
    num_epochs: int = 5,
    batch_size: int = 4
) -> str:
    """
    Train the model using the prepared training data.
    
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


def test_trained_model(model_path: str = "./ibani_model") -> None:
    """
    Test the trained model with some sample translations.
    
    Args:
        model_path: Path to the trained model directory
    """
    print("Testing trained model...")
    
    try:
        # Load the trained model
        translator = IbaniHuggingFaceTranslator(model_path=model_path)
        
        # Test sentences
        test_sentences = [
            "This is the genealogy of Jesus the Messiah",
            "Abraham was the father of Isaac",
            "I eat fish",
            "good morning",
            "thank you",
            "how are you",
            "I am fine"
        ]
        
        print("\nTranslation Results:")
        print("=" * 50)
        
        for sentence in test_sentences:
            try:
                translation = translator.translate(sentence)
                print(f"EN: {sentence}")
                print(f"IBANI: {translation}")
                print("-" * 30)
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
    """Main function to train model from ibani_eng.json."""
    print("Ibani Model Training from ibani_eng.json")
    print("=" * 50)
    
    # Step 1: Prepare training data
    print("\nStep 1: Preparing training data from ibani_eng.json...")
    try:
        training_examples = prepare_training_data_from_ibani_eng()
    except (FileNotFoundError, json.JSONDecodeError, IOError, ValueError) as e:
        print(f"Failed to prepare training data: {e}")
        return
    
    if len(training_examples) == 0:
        print("No training examples found! Exiting.")
        return
    
    # Step 2: Train the model
    print("\nStep 2: Training the model...")
    try:
        model_path = train_model_with_ibani_eng_data(
            training_data_file="ibani_eng_training_data.json",
            output_dir="./ibani_model",
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
    print("\nTraining pipeline completed!")
    print(f"Model saved to: {model_path}")
    print("The API will automatically use the new model from ./ibani_model")


if __name__ == "__main__":
    main()