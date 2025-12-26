"""
Simple script to translate English to Ibani using the trained model.
"""

from huggingface_translator import IbaniHuggingFaceTranslator


def main():
    """Load the trained model and translate sentences."""
    print("Loading trained Ibani translation model...")
    print("=" * 60)
    
    # Load your trained model
    translator = IbaniHuggingFaceTranslator(model_path="./ibani_model")
    
    print("✓ Model loaded successfully!\n")
    
    # Test sentences
    test_sentences = [
        "I eat fish",
        "The woman goes",
        "We see the man",
        "You drink water",
        "The child runs",
        "Good morning",
        "Thank you",
        "How are you",
        "I am fine",
        "I love you"
    ]
    
    print("Translating test sentences...")
    print("=" * 60)
    
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"EN:    {sentence}")
        print(f"IBANI: {translation}")
        print("-" * 60)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Translation Mode")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("Enter English text: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            # Skip empty input
            if not user_input:
                continue
            
            # Translate
            translation = translator.translate(user_input)
            print(f"Ibani translation: {translation}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()

