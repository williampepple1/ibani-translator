"""
Orthography mapping for Ibani special characters.
Focuses on the two problematic characters: ḅ and á
"""

# Only the characters that need mapping
IBANI_CHAR_MAP = {
    'ḅ': '[b_dot]',
    'Ḅ': '[B_dot]',
    'á': '[a_acute]',
    'Á': '[A_acute]',
}

# Reverse mapping
REVERSE_CHAR_MAP = {v: k for k, v in IBANI_CHAR_MAP.items()}


def encode_ibani_text(text: str) -> str:
    """Convert ḅ and á to safe placeholders before training."""
    result = text
    for char, placeholder in IBANI_CHAR_MAP.items():
        result = result.replace(char, placeholder)
    return result


def decode_ibani_text(text: str) -> str:
    """Convert placeholders back to ḅ and á after inference."""
    result = text
    for placeholder, char in REVERSE_CHAR_MAP.items():
        result = result.replace(placeholder, char)
    return result


def test_mapping():
    """Test the encoding/decoding."""
    test_texts = [
        "ḅẹlẹma árị",
        "Ebraham ḅara Jizọs",
        "This has ḅ and á characters",
    ]
    
    print("=" * 60)
    print("Testing ḅ and á Mapping")
    print("=" * 60)
    
    for text in test_texts:
        encoded = encode_ibani_text(text)
        decoded = decode_ibani_text(encoded)
        
        print(f"\nOriginal: {text}")
        print(f"Encoded:  {encoded}")
        print(f"Decoded:  {decoded}")
        print(f"Match:    {'✓' if text == decoded else '❌'}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_mapping()
