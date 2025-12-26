"""
Test script to demonstrate model loading from HuggingFace Hub.
This shows how the application loads models in different scenarios.
"""

import os
from huggingface_translator import IbaniHuggingFaceTranslator

def test_local_model():
    """Test loading from local directory."""
    print("=" * 70)
    print("TEST 1: Loading from Local Model Directory")
    print("=" * 70)
    
    if os.path.exists("./ibani_model"):
        translator = IbaniHuggingFaceTranslator(model_path="./ibani_model")
        test_translation(translator)
    else:
        print("❌ Local model not found at ./ibani_model")
        print("   Run train_from_ibani_eng.py first to create local model")
    print()

def test_huggingface_model():
    """Test loading from HuggingFace Hub."""
    print("=" * 70)
    print("TEST 2: Loading from HuggingFace Hub")
    print("=" * 70)
    
    # This will load from HuggingFace Hub since local path doesn't exist
    translator = IbaniHuggingFaceTranslator(
        model_path="./nonexistent_model",  # Intentionally wrong path
        hf_repo="williampepple1/ibani-translator"
    )
    test_translation(translator)
    print()

def test_with_environment_variables():
    """Test loading using environment variables (like Vercel deployment)."""
    print("=" * 70)
    print("TEST 3: Loading with Environment Variables (Production Mode)")
    print("=" * 70)
    
    # Set environment variables
    os.environ["HF_MODEL_REPO"] = "williampepple1/ibani-translator"
    os.environ["LOCAL_MODEL_PATH"] = "./ibani_model"
    
    # Get values from environment
    hf_repo = os.getenv("HF_MODEL_REPO")
    local_path = os.getenv("LOCAL_MODEL_PATH")
    
    print(f"Environment: HF_MODEL_REPO={hf_repo}")
    print(f"Environment: LOCAL_MODEL_PATH={local_path}")
    print()
    
    translator = IbaniHuggingFaceTranslator(
        model_path=local_path,
        hf_repo=hf_repo
    )
    test_translation(translator)
    print()

def test_translation(translator):
    """Test translation with sample sentences."""
    test_sentences = [
        "I am eating fish",
        "Good morning",
        "Thank you"
    ]
    
    print("\nTranslation Tests:")
    print("-" * 70)
    for sentence in test_sentences:
        try:
            translation = translator.translate(sentence)
            print(f"EN: {sentence}")
            print(f"IB: {translation}")
            print()
        except Exception as e:
            print(f"Error translating '{sentence}': {e}")
            print()

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("IBANI TRANSLATOR - MODEL LOADING TESTS")
    print("=" * 70)
    print("\nThis script demonstrates how the translator loads models:")
    print("1. From local directory (for development)")
    print("2. From HuggingFace Hub (for production/deployment)")
    print("3. Using environment variables (Vercel/production mode)")
    print()
    
    try:
        # Test 1: Local model
        test_local_model()
        
        # Test 2: HuggingFace Hub
        print("⚠️  Note: This will download the model from HuggingFace Hub")
        print("   Press Enter to continue or Ctrl+C to skip...")
        input()
        test_huggingface_model()
        
        # Test 3: Environment variables
        test_with_environment_variables()
        
        print("=" * 70)
        print("✓ All tests completed successfully!")
        print("=" * 70)
        print("\nDeployment Notes:")
        print("- On Vercel: Set HF_MODEL_REPO environment variable")
        print("- Local dev: Use ./ibani_model directory")
        print("- See DEPLOYMENT.md for detailed deployment instructions")
        print()
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during tests: {e}")

if __name__ == "__main__":
    main()

