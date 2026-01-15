"""
Hugging Face-based English to Ibani translator using MarianMT.
This implements training and inference for neural machine translation.
"""

import json
import torch
from transformers import (
    MarianMTModel, 
    MarianTokenizer, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import unicodedata
import re
from datasets import Dataset
from typing import List, Optional, Dict, Set
import os
from difflib import SequenceMatcher


class IbaniHuggingFaceTranslator:
    """
    English to Ibani neural machine translation using Hugging Face MarianMT.
    
    This class provides training and inference capabilities for translating
    English text to Ibani language using a fine-tuned MarianMT model.
    """
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-mul", 
                 model_path: Optional[str] = None,
                 hf_repo: Optional[str] = None,
                 training_data_file: str = "ibani_eng_training_data.json",
                 dictionary_file: str = "ibani_single_words.csv"):
        """
        Initialize the Hugging Face translator.
        
        Args:
            model_name: Pre-trained model to use as base
            model_path: Path to local fine-tuned model (if available)
            hf_repo: HuggingFace Hub repository (e.g., "username/ibani-translator")
                     Will be used if local model_path doesn't exist
            training_data_file: Path to training data for validation lookup
            dictionary_file: Path to the single words dictionary CSV
        """
        self.model_name = model_name
        self.model_path = model_path
        self.hf_repo = hf_repo
        
        # Load training data for anti-hallucination validation
        self.translation_lookup = {}
        self.en_to_ib_map = {}
        self.ib_to_en_map = {}
        self.known_en_words = set()
        self.known_ib_words = set()
        
        self._load_training_data_lookup(training_data_file)
        self._load_dictionary(dictionary_file)
        
        # Load tokenizer and model with fallback logic
        model_source = None
        
        # Try local path first
        if model_path and os.path.exists(model_path):
            print(f"✓ Loading fine-tuned model from local path: {model_path}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_path)
            self.model = MarianMTModel.from_pretrained(model_path)
            model_source = "local"
        
        # Try HuggingFace Hub if local doesn't exist
        elif hf_repo:
            try:
                print(f"Local model not found. Loading from HuggingFace Hub: {hf_repo}")
                self.tokenizer = MarianTokenizer.from_pretrained(hf_repo)
                self.model = MarianMTModel.from_pretrained(hf_repo)
                model_source = "huggingface"
                print(f"✓ Successfully loaded model from HuggingFace Hub")
            except Exception as e:
                print(f"Warning: Could not load from HuggingFace Hub: {e}")
                print(f"Falling back to base model: {model_name}")
                self.tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.model = MarianMTModel.from_pretrained(model_name)
                model_source = "base"
        
        # Fall back to base model
        else:
            print(f"Loading base pre-trained model: {model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            model_source = "base"
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✓ Using device: {self.device}")
        print(f"✓ Model source: {model_source}")
    
    def create_training_dataset(self, data_file: str = "training_data.json") -> Dataset:
        """Create training dataset from parallel English-Ibani data."""
        if not os.path.exists(data_file):
            print(f"Creating sample training data at {data_file}")
            self.create_sample_training_data(data_file)
        
        # Load the dataset
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to Hugging Face dataset format
        dataset = Dataset.from_list(data)
        return dataset
    
    def create_sample_training_data(self, output_file: str = "natural_training_data.json"):
        """Create basic sample training data with basic examples."""
        
        # Basic sample training data
        sample_data = [
            {"translation": {"en": "I eat fish", "ibani": "ịrị finji fíị"}},
            {"translation": {"en": "The woman goes", "ibani": "ọ́rụ́ḅọ́ má mú"}},
            {"translation": {"en": "We see the man", "ibani": "Wamini ówítụ́wọ ari"}},
            {"translation": {"en": "The child runs", "ibani": "Tuwo ma mangi"}}
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"Created training data with {len(sample_data)} basic examples")
    
    def preprocess_data(self, examples):
        """Preprocess data for training."""
        inputs = [self.normalize_text(ex["en"]) for ex in examples["translation"]]
        targets = [self.normalize_text(ex["ibani"]) for ex in examples["translation"]]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs, 
            max_length=128, 
            truncation=True, 
            padding=True
        )
        
        # Tokenize targets using the new method
        labels = self.tokenizer(
            text_target=targets,
            max_length=128, 
            truncation=True, 
            padding=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def train_model(self, 
                   training_data_file: str = "ibani_eng_training_data.json",
                   output_dir: str = "./ibani_model",
                   num_epochs: int = 10,
                   batch_size: int = 2,
                   learning_rate: float = 5e-5):
        """Train the model on English-Ibani data."""
        print("Starting model training...")
        
        # Create training dataset
        dataset = self.create_training_dataset(training_data_file)

        # Augment tokenizer with special characters from the dataset
        self._augment_tokenizer_from_data(dataset)
        
        # Preprocess the data
        tokenized_dataset = dataset.map(
            self.preprocess_data, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Use all data for training
        train_dataset = tokenized_dataset
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            # Use evaluation_strategy instead of eval_strategy for compatibility
            # eval_strategy="no",  # Disable evaluation for simplicity
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            save_total_limit=2,
            predict_with_generate=True,
            logging_steps=5,
            save_steps=50,
            warmup_steps=5,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            gradient_accumulation_steps=2,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        print("Training started...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model training completed! Model saved to {output_dir}")
        
        # Update model path for future use
        self.model_path = output_dir
    
    def _augment_tokenizer_from_data(self, dataset):
        """Add word fragments containing special characters to prevent spacing issues."""
        print("Checking for special characters and adding word fragments...")
        
        all_text = ""
        for ex in dataset:
            all_text += ex["translation"]["en"] + " " + ex["translation"]["ibani"]
        
        # Normalize text first
        all_text = self.normalize_text(all_text)
        
        # Find all unique characters
        unique_chars = set(all_text)
        
        # Characters to ignore (standard ASCII and common punctuation)
        ignore_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!:;()\"'- ")
        
        special_chars = [c for c in unique_chars if c not in ignore_chars and ord(c) > 127]
        
        if not special_chars:
            print("No special characters found.")
            return

        print(f"Found {len(special_chars)} special characters.")
        
        # CRITICAL: Instead of adding individual characters, add common word fragments
        # This prevents spacing issues
        vocab = self.tokenizer.get_vocab()
        
        # Priority characters that cause spacing issues
        priority_chars = ['ḅ', 'Ḅ', 'á', 'Á']
        
        # Collect word fragments containing these characters from the dataset
        word_fragments = set()
        
        for ex in dataset:
            ibani_text = ex["translation"]["ibani"]
            words = ibani_text.split()
            
            for word in words:
                # If word contains priority characters, add common fragments
                for char in priority_chars:
                    if char in word:
                        # Add the character with surrounding context
                        idx = word.find(char)
                        
                        # Add fragments like "ḅe", "ḅẹ", "aḅ", etc.
                        if idx > 0:
                            word_fragments.add(word[idx-1:idx+2])  # char with 1 before and 1 after
                        if idx < len(word) - 1:
                            word_fragments.add(word[idx:idx+2])    # char with 1 after
                        
                        # Also add the full word if it's short
                        if len(word) <= 8:
                            word_fragments.add(word)
        
        # Filter fragments not in vocabulary
        missing_fragments = [f for f in word_fragments if f not in vocab and len(f) > 1]
        
        if not missing_fragments:
            print("All word fragments already in vocabulary.")
            return
        
        # Limit to most common fragments (top 50)
        missing_fragments = sorted(missing_fragments)[:50]
        
        print(f"Adding {len(missing_fragments)} word fragments to tokenizer:")
        print(f"   Sample fragments: {missing_fragments[:10]}")
        
        num_added = self.tokenizer.add_tokens(missing_fragments)
        print(f"Added {num_added} tokens.")
        
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            print(f"Resized model embeddings to {len(self.tokenizer)}")

    def normalize_text(self, text: str) -> str:
        """Normalize text to NFC format to ensure tonal marks are consistent."""
        if not text:
            return text
        return unicodedata.normalize('NFC', text)
    
    def _load_training_data_lookup(self, training_data_file: str):
        """
        Load training data into a lookup dictionary for anti-hallucination validation.
        Creates a mapping of normalized English text to Ibani translations.
        Also builds word-level mappings for hallucination detection.
        """
        if not os.path.exists(training_data_file):
            print(f"Warning: Training data file '{training_data_file}' not found.")
            print("Anti-hallucination validation will be disabled.")
            return
        
        try:
            with open(training_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Build lookup dictionary with normalized keys
            for item in data:
                if "translation" in item:
                    en_text = item["translation"]["en"].strip()
                    ib_text = item["translation"]["ibani"].strip()
                    
                    norm_en = self.normalize_text(en_text.lower())
                    norm_ib = self.normalize_text(ib_text)
                    
                    self.translation_lookup[norm_en] = norm_ib
                    
                    # Basic word-level mapping from short sentences (1-3 words)
                    en_words = re.findall(r'\w+', en_text.lower())
                    ib_words = re.findall(r'\w+', ib_text.lower())
                    
                    for w in en_words: self.known_en_words.add(w)
                    for w in ib_words: self.known_ib_words.add(w)
                    
                    # If it's a very short sentence, we can assume mapping
                    if len(en_words) <= 2 and len(ib_words) <= 2:
                        for e_w in en_words:
                            for i_w in ib_words:
                                self.en_to_ib_map[e_w] = i_w
                                if i_w not in self.ib_to_en_map: self.ib_to_en_map[i_w] = set()
                                self.ib_to_en_map[i_w].add(e_w)
            
            print(f"✓ Loaded {len(self.translation_lookup)} translation pairs for validation")
        except Exception as e:
            print(f"Warning: Could not load training data: {e}")
            print("Anti-hallucination validation will be disabled.")

    def _load_dictionary(self, dictionary_file: str):
        """Load single word dictionary from CSV."""
        if not os.path.exists(dictionary_file):
            print(f"Warning: Dictionary file '{dictionary_file}' not found.")
            return
        
        try:
            import csv
            with open(dictionary_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # CSV format: Word (Ibani), Meaning (English)
                    ib_word = self.normalize_text(row['Word'].strip().lower())
                    en_meaning = self.normalize_text(row['Meaning'].strip().lower())
                    
                    # Meanings can have multiples, take first or clean
                    en_word = re.findall(r'\w+', en_meaning)[0] if re.findall(r'\w+', en_meaning) else en_meaning
                    
                    self.ib_to_en_map.setdefault(ib_word, set()).add(en_word)
                    self.en_to_ib_map[en_word] = ib_word
                    self.known_en_words.add(en_word)
                    self.known_ib_words.add(ib_word)
            print(f"✓ Loaded {len(self.ib_to_en_map)} words from dictionary")
        except Exception as e:
            print(f"Warning: Could not load dictionary: {e}")
    
    def _find_best_match(self, text: str, threshold: float = 0.85) -> Optional[str]:
        """
        Find the best matching translation from training data.
        
        Args:
            text: Normalized English text to match
            threshold: Minimum similarity score (0-1) to consider a match
        
        Returns:
            Ibani translation if a good match is found, None otherwise
        """
        if not self.translation_lookup:
            return None
        
        normalized_text = text.strip().lower()
        
        # First try exact match
        if normalized_text in self.translation_lookup:
            return self.translation_lookup[normalized_text]
        
        # Try fuzzy matching for close matches
        best_match = None
        best_score = 0.0
        
        for key, value in self.translation_lookup.items():
            # Calculate similarity
            similarity = SequenceMatcher(None, normalized_text, key).ratio()
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = value
        
        return best_match

    def translate(self, text: str, use_validation: bool = True) -> str:
        """
        Translate English text to Ibani.
        
        Args:
            text: English text to translate
            use_validation: If True, validate translation against training data
                          and return original text if no match found
        """
        if not text.strip():
            return ""

        # Normalize input
        original_text = text
        text = self.normalize_text(text)
        
        # 1. Full Sentence Lookup (High Priority)
        if use_validation and self.translation_lookup:
            validated_translation = self._find_best_match(text)
            if validated_translation:
                return validated_translation

        # 2. Neural Translation with Token-Level Hallucination Filter
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = self.normalize_text(translation)
        
        if not use_validation:
            return translation

        # 3. Token-Level Validation (Hallucination Detection)
        # If the model output contains Ibani words that map to English words NOT in the input,
        # it's likely a hallucination of an OOV word.
        en_tokens = set(re.findall(r'\w+', text.lower()))
        ib_tokens = re.findall(r'\w+', translation) # Keep case for structural tokens maybe, but processing usually lower
        
        final_tokens = []
        # We need to preserve punctuation and structure as much as possible, 
        # so we'll do a simple split and replace.
        words_in_output = translation.split()
        
        for word in words_in_output:
            clean_word = re.sub(r'[^\w]', '', word).lower()
            if not clean_word:
                final_tokens.append(word)
                continue
                
            # Check if this Ibani word corresponds to a different English word
            meanings = self.ib_to_en_map.get(clean_word, set())
            
            is_hallucination = False
            if meanings:
                # If ALL possible English meanings for this Ibani word are NOT in the input,
                # then this Ibani word is likely a hallucination of something else.
                if not any(m in en_tokens for m in meanings):
                    is_hallucination = True
            
            if is_hallucination:
                # Try to find which English word from the input was likely meant to be here.
                # A simple heuristic: find an English word in input that isn't known to be elsewhere in output.
                # But to keep it simple as requested: "non-existent words should show in original form"
                # We'll try to find an input word that is NOT in known_en_words (an OOV word)
                oov_input_words = [w for w in en_tokens if w not in self.known_en_words]
                if oov_input_words:
                    # Replace with the first OOV word found (or a better heuristic)
                    # For "The Jet", "Jet" is OOV. If output is "Man ma", "Man" is hallucination.
                    # We replace "Man" with "Jet".
                    final_tokens.append(word.replace(re.sub(r'[^\w]', '', word), oov_input_words[0]))
                else:
                    final_tokens.append(word)
            else:
                final_tokens.append(word)

        return " ".join(final_tokens)
    
    def batch_translate(self, texts: List[str]) -> List[str]:
        """Translate multiple texts at once."""
        translations = []
        for text in texts:
            translation = self.translate(text)
            translations.append(translation)
        return translations


def main():
    """Example usage of the Hugging Face translator."""
    print("Hugging Face English to Ibani Translator")
    print("=" * 50)
    
    # Initialize translator
    translator = IbaniHuggingFaceTranslator()
    
    # Test sentences
    test_sentences = [
        "I eat fish",
        "The woman goes",
        "We see the man",
        "You drink water",
        "The child runs"
    ]
    
    print("Testing pre-trained model (English-Swahili base):")
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"English: {sentence}")
        print(f"Translation: {translation}")
        print()
    
    # Automatically create training data and start training
    print("Creating training data and starting model training...")
    translator.create_sample_training_data("training_data.json")
    translator.train_model()
    
    # Test with trained model
    print("\nTesting with fine-tuned model:")
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"English: {sentence}")
        print(f"Ibani: {translation}")
        print()


if __name__ == "__main__":
    main()
