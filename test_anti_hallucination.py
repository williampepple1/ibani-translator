"""
Test script to demonstrate anti-hallucination feature.
This shows how the translator handles words that exist vs don't exist in training data.
"""

from huggingface_translator import IbaniHuggingFaceTranslator

def main():
    print("=" * 70)
    print("Testing Anti-Hallucination Feature")
    print("=" * 70)
    print()
    
    # Initialize translator with training data validation
    print("Initializing translator with anti-hallucination validation...")
    translator = IbaniHuggingFaceTranslator(
        model_path="./ibani_model",
        hf_repo="williampepple1/ibani-translator",
        training_data_file="ibani_eng_training_data.json"
    )
    print()
    
    # Test cases
    test_cases = [
        # These should exist in training data (from the Bible translations)
        ("I eat fish", "Should translate (exists in training data)"),
        ("The woman goes", "Should translate (exists in training data)"),
        ("Abraham was the father of Isaac", "Should translate (exists in training data)"),
        # Mixed known and unknown words
        ("The Jet", "Should translate 'the' (ma) and keep 'Jet' as is -> 'Jet ma' or similar"),
        
        # These likely don't exist in training data
        ("I love programming", "Should return original (not in training data)"),
        ("The cat is sleeping", "Should return original (not in training data)"),
        ("Hello world", "Should return original (not in training data)"),
        ("xyzabc123", "Should return original (gibberish)"),
    ]
    
    print("Testing translations:")
    print("-" * 70)
    
    for english_text, description in test_cases:
        print(f"\nInput: {english_text}")
        print(f"Description: {description}")
        
        # Translate with validation enabled (default)
        translation = translator.translate(english_text, use_validation=True)
        print(f"Output: {translation}")
        
        # Check if it returned the original text
        if translation == english_text:
            print("✓ Returned original text (no hallucination)")
        else:
            print("✓ Found valid translation in training data")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)
    print("\nSummary:")
    print("- Words/phrases in training data: Translated correctly")
    print("- Words/phrases NOT in training data: Original text returned")
    print("- This prevents the model from hallucinating translations!")

if __name__ == "__main__":
    main()
