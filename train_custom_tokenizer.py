"""
Train a custom SentencePiece tokenizer for Ibani language.
This tokenizer will include ALL Ibani characters (ḅ, á, ọ́, etc.)
and can be used with MarianMT for faster training/inference than ByT5.
"""

import sentencepiece as spm
import json
import csv
from pathlib import Path


def extract_ibani_text_from_csv(csv_file: str = "ibani_eng.csv") -> str:
    """
    Extract all Ibani text from CSV to create tokenizer training corpus.
    """
    print(f"Extracting Ibani text from {csv_file}...")
    
    ibani_texts = []
    english_texts = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            ibani = row.get("ibani_text", "").strip()
            english = row.get("nlt_text", "").strip()
            
            if ibani:
                ibani_texts.append(ibani)
            if english:
                english_texts.append(english)
    
    print(f"Extracted {len(ibani_texts)} Ibani sentences")
    print(f"Extracted {len(english_texts)} English sentences")
    
    # Combine both for a bilingual tokenizer
    all_text = "\n".join(ibani_texts + english_texts)
    
    return all_text, ibani_texts, english_texts


def train_sentencepiece_tokenizer(
    text_data: str,
    model_prefix: str = "ibani_tokenizer",
    vocab_size: int = 8000,
    character_coverage: float = 1.0
):
    """
    Train a SentencePiece tokenizer that preserves ALL characters.
    
    Args:
        text_data: Training text corpus
        model_prefix: Output model name prefix
        vocab_size: Vocabulary size (8000 is good for small datasets)
        character_coverage: 1.0 = include ALL characters (important!)
    """
    print(f"\nTraining SentencePiece tokenizer...")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Character coverage: {character_coverage} (100% - all characters preserved)")
    
    # Save text to temporary file
    corpus_file = "tokenizer_corpus.txt"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write(text_data)
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,  # 1.0 = include ALL characters
        model_type='unigram',  # unigram is best for low-resource languages
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[],  # Can add special tokens here
        normalization_rule_name='nfkc',  # Normalize Unicode
        byte_fallback=True,  # Important: handle unknown bytes
    )
    
    print(f"✓ Tokenizer saved to {model_prefix}.model and {model_prefix}.vocab")
    
    # Clean up
    Path(corpus_file).unlink()
    
    return f"{model_prefix}.model"


def test_tokenizer(model_file: str, test_sentences: list):
    """Test the trained tokenizer."""
    print(f"\n{'='*70}")
    print("Testing Tokenizer")
    print(f"{'='*70}")
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    
    for sentence in test_sentences:
        # Encode
        pieces = sp.encode_as_pieces(sentence)
        ids = sp.encode_as_ids(sentence)
        
        # Decode
        decoded = sp.decode_pieces(pieces)
        
        print(f"\nOriginal:  {sentence}")
        print(f"Pieces:    {pieces}")
        print(f"IDs:       {ids[:10]}..." if len(ids) > 10 else f"IDs:       {ids}")
        print(f"Decoded:   {decoded}")
        
        # Check for special characters
        special_chars = ['ḅ', 'á', 'ọ', 'ẹ', 'í', 'ú', 'ó']
        found = [c for c in special_chars if c in sentence]
        preserved = [c for c in found if c in decoded]
        
        if found:
            if len(preserved) == len(found):
                print(f"✓ All special characters preserved: {', '.join(found)}")
            else:
                print(f"❌ Lost characters: {set(found) - set(preserved)}")


def create_tokenizer_config(model_prefix: str = "ibani_tokenizer"):
    """Create configuration file for the tokenizer."""
    config = {
        "model_file": f"{model_prefix}.model",
        "vocab_file": f"{model_prefix}.vocab",
        "vocab_size": 8000,
        "character_coverage": 1.0,
        "model_type": "unigram",
        "description": "Custom SentencePiece tokenizer for Ibani language with full Unicode support",
        "special_characters": ["ḅ", "Ḅ", "á", "Á", "ọ", "Ọ", "ẹ", "Ẹ", "í", "Í", "ú", "Ú", "ó", "Ó"],
        "usage": "Use with MarianMT or any Transformer model"
    }
    
    config_file = f"{model_prefix}_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Configuration saved to {config_file}")


def main():
    """Main function to train custom Ibani tokenizer."""
    print("="*70)
    print("Custom SentencePiece Tokenizer Training for Ibani")
    print("="*70)
    print("This tokenizer will preserve ALL Ibani characters (ḅ, á, ọ́, etc.)")
    print("Can be used with MarianMT for faster training than ByT5")
    print("="*70)
    
    # Step 1: Extract text from CSV
    print("\nStep 1: Extracting text from CSV...")
    all_text, ibani_texts, english_texts = extract_ibani_text_from_csv()
    
    # Step 2: Train tokenizer
    print("\nStep 2: Training SentencePiece tokenizer...")
    model_file = train_sentencepiece_tokenizer(
        text_data=all_text,
        model_prefix="ibani_tokenizer",
        vocab_size=8000,
        character_coverage=1.0  # CRITICAL: Include ALL characters
    )
    
    # Step 3: Test tokenizer
    print("\nStep 3: Testing tokenizer with Ibani sentences...")
    test_sentences = [
        # From your CSV data
        "Mịị anịị diri bie anị fịnị ḅara Jizọs tádọ́apụ",
        "Ebraham anịị Aizik daa, Aizik anịị Jekọpụ daa",
        "Juda anịị Pẹrẹzi na Zẹra daa, Táma anịị nna nyingi",
        # English
        "Abraham was the father of Isaac",
        "This is the genealogy of Jesus"
    ]
    
    test_tokenizer(model_file, test_sentences)
    
    # Step 4: Create config
    print("\nStep 4: Creating configuration file...")
    create_tokenizer_config()
    
    print("\n" + "="*70)
    print("Tokenizer Training Complete!")
    print("="*70)
    print("\nFiles created:")
    print("  - ibani_tokenizer.model (SentencePiece model)")
    print("  - ibani_tokenizer.vocab (Vocabulary)")
    print("  - ibani_tokenizer_config.json (Configuration)")
    print("\nNext steps:")
    print("  1. Use this tokenizer with MarianMT")
    print("  2. Train MarianMT with the custom tokenizer")
    print("  3. Enjoy faster inference than ByT5!")
    print("="*70)


if __name__ == "__main__":
    # Check if sentencepiece is installed
    try:
        import sentencepiece
        main()
    except ImportError:
        print("ERROR: sentencepiece not installed")
        print("Install it with: pip install sentencepiece")
