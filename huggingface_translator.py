"""
Hugging Face-based English to Ibani translator using MarianMT.
This implements training and inference for neural machine translation.
"""

import json
import os
import re
import unicodedata
from typing import List, Optional, Dict

import torch
from transformers import (
    MarianMTModel, 
    MarianTokenizer, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset


class IbaniHuggingFaceTranslator:
    """
    English to Ibani neural machine translation using Hugging Face MarianMT.
    
    This class provides training and inference capabilities for translating
    English text to Ibani language using a fine-tuned MarianMT model.
    """
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-mul", 
                 model_path: Optional[str] = None,
                 hf_repo: Optional[str] = None):
        """
        Initialize the Hugging Face translator.
        
        Args:
            model_name: Pre-trained model to use as base
            model_path: Path to local fine-tuned model (if available)
            hf_repo: HuggingFace Hub repository (e.g., "username/ibani-translator")
                     Will be used if local model_path doesn't exist
        """
        self.model_name = model_name
        self.model_path = model_path
        self.hf_repo = hf_repo
        
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
        """Add missing Ibani characters (á and ḅ) to tokenizer."""
        print("Augmenting tokenizer with missing Ibani characters...")
        
        # Only the specific characters that don't show up in output
        ibani_special_chars = ['á', 'Á', 'ḅ', 'Ḅ']
        
        vocab = self.tokenizer.get_vocab()
        
        # Add missing characters
        missing_chars = [c for c in ibani_special_chars if c not in vocab]
        
        if missing_chars:
            print(f"Adding {len(missing_chars)} missing characters: {missing_chars}")
            num_added = self.tokenizer.add_tokens(missing_chars)
            print(f"Added {num_added} character tokens.")
            
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"Resized model embeddings to {len(self.tokenizer)}")
        else:
            print("Characters á and ḅ already in vocabulary.")
        
        # Add common word fragments containing these characters
        word_fragments = set()
        
        for ex in dataset:
            ibani_text = self.normalize_text(ex["translation"]["ibani"])
            words = ibani_text.split()
            
            for word in words:
                for char in ibani_special_chars:
                    if char in word:
                        idx = word.find(char)
                        # Add fragments with context
                        if idx > 0 and idx < len(word) - 1:
                            word_fragments.add(word[idx-1:idx+2])
                        if idx < len(word) - 1:
                            word_fragments.add(word[idx:idx+2])
                        # Add short words containing the character
                        if len(word) <= 6:
                            word_fragments.add(word)
        
        # Get current vocab after adding characters
        vocab = self.tokenizer.get_vocab()
        missing_fragments = [f for f in word_fragments if f not in vocab and len(f) > 1]
        
        if missing_fragments:
            missing_fragments = sorted(missing_fragments)[:50]
            print(f"Adding {len(missing_fragments)} word fragments containing á/ḅ")
            
            num_added = self.tokenizer.add_tokens(missing_fragments)
            print(f"Added {num_added} fragment tokens.")
            
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"Final vocabulary size: {len(self.tokenizer)}")
        
        print("Tokenizer augmentation complete.")

    def normalize_text(self, text: str) -> str:
        """Normalize text to NFC format to ensure tonal marks are consistent."""
        if not text:
            return text
        return unicodedata.normalize('NFC', text)
    
    def fix_ibani_spacing(self, text: str) -> str:
        """Remove unwanted spaces around special Ibani characters (á, ḅ)."""
        if not text:
            return text
        
        # Characters that should not have spaces around them when part of a word
        special_chars = ['á', 'Á', 'ḅ', 'Ḅ']
        
        for char in special_chars:
            # Remove space before the character (when preceded by a letter)
            text = re.sub(r'([a-zA-Zạẹịọụ])\s+' + re.escape(char), r'\1' + char, text)
            # Remove space after the character (when followed by a letter)
            text = re.sub(re.escape(char) + r'\s+([a-zA-Zạẹịọụ])', char + r'\1', text)
        
        return text
    
    def translate(self, text: str) -> str:
        """
        Translate English text to Ibani.
        
        Args:
            text: English text to translate
        """
        if not text.strip():
            return ""

        # Normalize input
        text = self.normalize_text(text)
        
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
                no_repeat_ngram_size=3 # Prevent some types of gibberish
            )
        
        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = self.normalize_text(translation)
        translation = self.fix_ibani_spacing(translation)
        
        return translation
    
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
