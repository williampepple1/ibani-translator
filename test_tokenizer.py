
from huggingface_translator import IbaniHuggingFaceTranslator

def test_tokenizer():
    translator = IbaniHuggingFaceTranslator()
    text = "ḅara ạna áká ọ́rụ́ḅọ́"
    
    print(f"Original: {text}")
    tokens = translator.tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    
    ids = translator.tokenizer.convert_tokens_to_ids(tokens)
    print(f"IDs: {ids}")
    
    decoded = translator.tokenizer.decode(ids)
    print(f"Decoded: {decoded}")

if __name__ == "__main__":
    test_tokenizer()
