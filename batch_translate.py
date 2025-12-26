"""
Batch translate English sentences to Ibani from a text file.
"""

import sys
from huggingface_translator import IbaniHuggingFaceTranslator


def batch_translate_file(input_file: str, output_file: str, model_path: str = "./ibani_model"):
    """
    Translate all lines from input file and save to output file.
    
    Args:
        input_file: Path to file with English sentences (one per line)
        output_file: Path to save Ibani translations
        model_path: Path to the trained model
    """
    print(f"Loading model from {model_path}...")
    translator = IbaniHuggingFaceTranslator(model_path=model_path)
    print("✓ Model loaded!\n")
    
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print(f"Translating {len(sentences)} sentences...")
    print("=" * 60)
    
    # Translate and save
    translations = []
    for i, sentence in enumerate(sentences, 1):
        try:
            translation = translator.translate(sentence)
            translations.append(translation)
            print(f"[{i}/{len(sentences)}] {sentence} → {translation}")
        except Exception as e:
            print(f"[{i}/{len(sentences)}] Error translating '{sentence}': {e}")
            translations.append("[ERROR]")
    
    # Save to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in translations:
                f.write(translation + '\n')
        print(f"\n✓ Translations saved to {output_file}")
    except Exception as e:
        print(f"\nError saving to '{output_file}': {e}")


def main():
    """Main function for batch translation."""
    if len(sys.argv) < 3:
        print("Usage: python batch_translate.py <input_file> <output_file>")
        print("\nExample:")
        print("  python batch_translate.py english.txt ibani.txt")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    batch_translate_file(input_file, output_file)


if __name__ == "__main__":
    main()

