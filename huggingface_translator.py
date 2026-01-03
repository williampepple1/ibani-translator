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
from datasets import Dataset
from typing import List, Optional
import os


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
        """Create comprehensive training data using the rule-based translator."""
        # Create extensive training examples
        training_sentences = [
            # Basic sentences
            "I eat fish", "I ate fish", "I will eat fish", "I have eaten fish", "I am eating fish",
            "The woman goes", "The woman went", "The woman will go", "The woman has gone", "The woman is going",
            "We see the man", "We saw the man", "We will see the man", "We have seen the man", "We are seeing the man",
            "You drink water", "You drank water", "You will drink water", "You have drunk water", "You are drinking water",
            "The child runs", "The child ran", "The child will run", "The child has run", "The child is running",
            
            # SOV examples
            "The man slapped me", "The man will slap me", "The man has slapped me",
            "The child sees the woman", "The child saw the woman", "The child will see the woman",
            "I see you", "I saw you", "I will see you",
            "The woman eats fish", "The woman ate fish", "The woman will eat fish",
            
            # More complex sentences
            "The house is big", "The house was big", "The house will be big",
            "I love you", "I loved you", "I will love you",
            "Good morning", "Thank you", "How are you", "I am fine",
            "What is your name", "My name is John", "Where are you going",
            "I am going home", "The sun is hot", "The water is cold",
            
            # Additional examples for better training
            "The dog barks", "The dog barked", "The dog will bark",
            "The cat sleeps", "The cat slept", "The cat will sleep",
            "The bird flies", "The bird flew", "The bird will fly",
            "The fish swims", "The fish swam", "The fish will swim",
            "The tree grows", "The tree grew", "The tree will grow"
        ]
        
        # Use rule-based translator to generate Ibani translations
        try:
            from rule_based_translator import IbaniRuleBasedTranslator
            rule_translator = IbaniRuleBasedTranslator()
            
            sample_data = []
            for sentence in training_sentences:
                ibani_translation = rule_translator.translate_sentence(sentence)
                sample_data.append({
                    "translation": {
                        "en": sentence,
                        "ibani": ibani_translation
                    }
                })
            
            print(f"Generated {len(sample_data)} training examples using rule-based translator")
            
        except (ImportError, AttributeError, ValueError, RuntimeError) as e:
            print(f"Error using rule-based translator: {e}")
            # Fallback to basic examples
            sample_data = [
                {"translation": {"en": "I eat fish", "ibani": "ịrị olokpó fíị"}},
                {"translation": {"en": "The woman goes", "ibani": "ọ́rụ́ḅọ́ má mú"}},
                {"translation": {"en": "We see the man", "ibani": "ami ówítụ́wọ ari"}},
                {"translation": {"en": "You drink water", "ibani": "wori min na"}},
                {"translation": {"en": "The child runs", "ibani": "tamuno mangi"}}
            ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"Created training data with {len(sample_data)} examples using comprehensive dictionary")
    
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
                   training_data_file: str = "natural_training_data.json",
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
    

    def translate(self, text: str, use_fallback: bool = False) -> str:
        """
        Translate English text to Ibani.
        
        Args:
            text: English text to translate
            use_fallback: Whether to use rule-based fallback for unknown words
        """
        if not text.strip():
            return ""

        # Normalize input
        text = self.normalize_text(text)
        
        # Check if the input is a single word and exists in rule-based dictionary
        # This helps with the "meaningless words" problem for unknown English words
        if use_fallback:
            try:
                from rule_based_translator import IbaniRuleBasedTranslator
                rb_translator = IbaniRuleBasedTranslator()
                
                # If it's a very short text, rule-based might be more reliable if ML model fails
                is_single_word = len(text.split()) == 1
                if is_single_word:
                    # Clean the word
                    import re
                    clean_word = re.sub(r'[^\w]', '', text.lower())
                    
                    # If the word is in the dictionary, we might want to keep it in mind
                    # but let's see what the model does first.
                    pass
            except Exception:
                rb_translator = None
        else:
            rb_translator = None

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

        # Post-processing: Check if the translation looks "meaningless"
        # 1. Hallucination check: If external model source is 'base', it might be Swahili!
        # 2. Unknown word check: If the translation is identical to the source or looks like gibberish
        if use_fallback and rb_translator:
            # If the model produced something that looks like it's from the base model (Swahili-like)
            # or if it's very different from the dictionary for simple inputs
            if len(text.split()) <= 3:
                rb_translation = rb_translator.translate_sentence(text)
                # If model output is very short or looks like Swahili (base model)
                # and we have a rule-base translation, we might prefer the rule-base
                # for very short common phrases
                if translation == text or not translation:
                    return rb_translation
        
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
