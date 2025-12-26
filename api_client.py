"""
Example client for using the Ibani Translation API.
"""

import requests
import json


# API base URL
API_URL = "http://localhost:8080"


def translate_text(text: str) -> dict:
    """
    Translate a single text using the API.
    
    Args:
        text: English text to translate
        
    Returns:
        Dictionary with translation result
    """
    response = requests.post(
        f"{API_URL}/translate",
        json={"text": text}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def batch_translate(texts: list) -> dict:
    """
    Translate multiple texts using the API.
    
    Args:
        texts: List of English texts to translate
        
    Returns:
        Dictionary with batch translation results
    """
    response = requests.post(
        f"{API_URL}/batch-translate",
        json={"texts": texts}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def check_health() -> dict:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json()
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API. Is the server running?")
        return None


def main():
    """Example usage of the API client."""
    print("Ibani Translation API Client")
    print("=" * 60)
    
    # Check API health
    print("\n1. Checking API health...")
    health = check_health()
    if health:
        print(f"✓ API Status: {health['status']}")
        print(f"✓ Model Loaded: {health['model_loaded']}")
    else:
        print("✗ API is not available. Start the server with: python api_server.py")
        return
    
    # Single translation
    print("\n2. Single Translation Example:")
    print("-" * 60)
    text = "I eat fish"
    result = translate_text(text)
    if result:
        print(f"English: {result['source']}")
        print(f"Ibani:   {result['translation']}")
    
    # Batch translation
    print("\n3. Batch Translation Example:")
    print("-" * 60)
    texts = [
        "Good morning",
        "Thank you",
        "How are you",
        "I love you",
        "The woman goes"
    ]
    
    results = batch_translate(texts)
    if results:
        print(f"Translated {results['count']} sentences:")
        for item in results['translations']:
            print(f"  EN:    {item['source']}")
            print(f"  IBANI: {item['translation']}")
            print()
    
    # Interactive mode
    print("\n4. Interactive Mode")
    print("-" * 60)
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter English text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            result = translate_text(user_input)
            if result:
                print(f"Ibani: {result['translation']}\n")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break


if __name__ == "__main__":
    main()

