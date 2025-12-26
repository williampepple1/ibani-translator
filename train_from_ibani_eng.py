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
    output_file: str = "ibani_eng_training_data.json",
    dictionary_file: str = "ibani_dict.json"
) -> List[Dict[str, Dict[str, str]]]:
    """
    Extract data and augment with dictionary for better coverage.
    """
    print(f"Loading data from {input_file}...")
    training_examples = []
    
    # Load primary data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                en = entry.get("english_text", "").strip()
                ib = entry.get("ibani_text", "").strip()
                if en and ib:
                    training_examples.append({"translation": {"en": en, "ibani": ib}})
    except Exception as e:
        print(f"Warning: Could not load primary data: {e}")

    # Load dictionary data for augmentation (REMOVED as per user request to not use it as fallback)
    # try:
    #     print(f"Augmenting with dictionary from {dictionary_file}...")
    #     with open(dictionary_file, 'r', encoding='utf-8') as f:
    #         dict_data = json.load(f)
    #         for entry in dict_data:
    #             en = entry.get("word", "").strip()
    #             ib = entry.get("Ibani_word", "").strip()
    #             if en and ib and "," not in en:  # Simple 1-to-1 mappings
    #                 training_examples.append({"translation": {"en": en, "ibani": ib}})
    # except Exception as e:
    #     print(f"Warning: Could not load dictionary: {e}")

    # Add identity mappings for common names and English words (Copy Task)
    # This helps the model realize some words shouldn't be "translated" to garbage
    common_identities = ["John", "Mary", "Jesus", "David", "Abraham", "Peter", "Paul", "Lagos", "Nigeria", "Internet", "Computer"]
    for word in common_identities:
        training_examples.append({"translation": {"en": word, "ibani": word}})

    print(f"Total training examples: {len(training_examples)}")
    
    # Save training data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, ensure_ascii=False, indent=2)
    
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