"""
Orthography mapping for Ibani special characters.
Focuses on ḅ and á with aggressive pattern matching for corrupted outputs.
"""

import re

# Encoding map (used before training)
IBANI_CHAR_MAP = {
    'ḅ': '[b_dot]',
    'Ḅ': '[B_dot]',
    'á': '[a_acute]',
    'Á': '[A_acute]',
}

REVERSE_CHAR_MAP = {v: k for k, v in IBANI_CHAR_MAP.items()}


def encode_ibani_text(text: str) -> str:
    """Convert ḅ and á to safe placeholders before training."""
    result = text
    for char, placeholder in IBANI_CHAR_MAP.items():
        result = result.replace(char, placeholder)
    return result


def decode_ibani_text(text: str) -> str:
    """
    Convert placeholders back to ḅ and á after inference.
    Handles corrupted/partial placeholders from model output.
    """
    result = text
    
    # Pattern 1: Standard and known variations for ḅ
    b_patterns = [
        r'\[b_dot\]',
        r'\[b_dom\]',
        r'\[b_do\]',
        r'\[b_d\]',
        r'\[b_\]',
        r'\[B_dot\]',
        r'\[B_dom\]',
    ]
    
    # Pattern 2: Standard and known variations for á  
    a_patterns = [
        r'\[a_acute\]',
        r'\[a_acut\]',
        r'\[a_acu\]',
        r'\[a_ac\]',
        r'\[a_a\]',
        r'\[a_\]',
        r'\[a_im\[a',  # Your specific corrupted pattern
        r'\[A_acute\]',
        r'\[A_acut\]',
    ]
    
    # Apply ḅ patterns
    for pattern in b_patterns:
        result = re.sub(pattern, 'ḅ', result, flags=re.IGNORECASE)
    
    # Apply á patterns
    for pattern in a_patterns:
        result = re.sub(re.escape(pattern) if '[' in pattern and pattern.count('[') > 1 else pattern, 
                       'á', result, flags=re.IGNORECASE)
    
    # Catch-all: Any remaining [x_...] patterns that likely should be á or ḅ
    # Pattern: [letter_ followed by anything until ]
    result = re.sub(r'\[a_[^\]]*\]', 'á', result, flags=re.IGNORECASE)
    result = re.sub(r'\[b_[^\]]*\]', 'ḅ', result, flags=re.IGNORECASE)
    
    # Clean up broken patterns like [a_im[a (unclosed brackets)
    result = re.sub(r'\[a_[a-z]*\[a', 'á', result, flags=re.IGNORECASE)
    result = re.sub(r'\[b_[a-z]*\[b', 'ḅ', result, flags=re.IGNORECASE)
    
    # Remove any remaining orphan brackets from corrupted outputs
    result = re.sub(r'\[[ab]_[^\]]*$', '', result)  # Unclosed at end
    result = re.sub(r'\[_', '', result)  # Broken starts
    
    return result


def test_mapping():
    """Test the encoding/decoding with real corrupted outputs."""
    
    # Corrupted outputs from your model
    corrupted_outputs = [
        "ám[a_im[a",  # Should be: ámányánáḅọ (your example)
        "[b_dom]ẹẹ",  # Should be: ḅẹẹ
        "Iyẹ Tamuno, a ịmiẹḅam pa ọmịna [b_dom]ẹẹ.",
        "[a_acute]rị [b_dot]ẹlẹma",
        "[a_acu]rị",
        "[a_]test",
        "[b_]test",
    ]
    
    print("=" * 60)
    print("Testing Corrupted Pattern Decoding")
    print("=" * 60)
    
    for output in corrupted_outputs:
        decoded = decode_ibani_text(output)
        has_bracket = '[' in decoded
        print(f"  Input:   {output}")
        print(f"  Output:  {decoded}")
        print(f"  Clean:   {'❌' if has_bracket else '✓'}\n")
    
    print("=" * 60)


if __name__ == "__main__":
    test_mapping()
