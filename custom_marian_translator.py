"""
Train MarianMT with custom SentencePiece tokenizer for Ibani.
This combines the speed of MarianMT with perfect character preservation.
Uses a simple SentencePiece wrapper instead of PreTrainedTokenizerFast.
"""

import json
import torch
from transformers import (
    MarianMTModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from typing import List, Dict, Optional
import os
import sentencepiece as spm
import numpy as np


class SentencePieceTokenizerWrapper:
    """Simple wrapper around SentencePiece for HuggingFace compatibility."""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Special tokens
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        
        # Token IDs
        self.pad_token_id = self.sp.piece_to_id(self.pad_token)
        self.eos_token_id = self.sp.piece_to_id(self.eos_token)
        self.unk_token_id = self.sp.piece_to_id(self.unk_token)
        self.bos_token_id = self.sp.piece_to_id(self.bos_token)
        
        self.model_max_length = 512
    
    def __len__(self):
        return self.sp.get_piece_size()
    
    def __call__(self, text=None, max_length=128, truncation=True, padding=False, 
                 return_tensors=None, text_target=None):
        """Tokenize text (compatible with HuggingFace interface)."""
        
        # Use text_target if provided (for labels), otherwise use text
        input_text = text_target if text_target is not None else text
        
        if input_text is None:
            raise ValueError("Either 'text' or 'text_target' must be provided")
        
        # Handle batch input
        if isinstance(input_text, list):
            texts = input_text
        else:
            texts = [input_text]
        
        # Encode
        input_ids = []
        for t in texts:
            ids = self.sp.encode_as_ids(t)
            # Truncate
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            # Pad
            if padding and len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            input_ids.append(ids)
        
        # Create attention mask
        attention_mask = [[1 if id != self.pad_token_id else 0 for id in ids] 
                         for ids in input_ids]
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"])
            result["attention_mask"] = torch.tensor(result["attention_mask"])
        
        return result
    
    def encode(self, text, add_special_tokens=True):
        """Encode text to IDs."""
        return self.sp.encode_as_ids(text)
    
    def decode(self, token_ids, skip_special_tokens=True):
        """Decode IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Remove padding if skip_special_tokens
        if skip_special_tokens:
            token_ids = [id for id in token_ids if id not in [self.pad_token_id, self.eos_token_id, self.bos_token_id]]
        
        return self.sp.decode_ids(token_ids)
    
    def batch_decode(self, sequences, skip_special_tokens=True):
        """Decode batch of sequences."""
        return [self.decode(seq, skip_special_tokens) for seq in sequences]
    
    def pad(self, encoded_inputs, padding=True, max_length=None, return_tensors=None):
        """Pad encoded inputs (required by DataCollator)."""
        # Handle dict or list of dicts
        if isinstance(encoded_inputs, dict):
            # Single example
            batch = [encoded_inputs]
        else:
            batch = encoded_inputs
        
        # Get max length
        if max_length is None:
            max_length = max(len(item["input_ids"]) for item in batch)
        
        # Pad each item
        padded_batch = {
            "input_ids": [],
            "attention_mask": []
        }
        
        if "labels" in batch[0]:
            padded_batch["labels"] = []
        
        for item in batch:
            # Pad input_ids
            input_ids = item["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            
            padded_ids = input_ids + [self.pad_token_id] * (max_length - len(input_ids))
            padded_batch["input_ids"].append(padded_ids)
            
            # Pad attention_mask
            if "attention_mask" in item:
                attention_mask = item["attention_mask"]
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.tolist()
                padded_mask = attention_mask + [0] * (max_length - len(attention_mask))
            else:
                padded_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            padded_batch["attention_mask"].append(padded_mask)
            
            # Pad labels if present
            if "labels" in item:
                labels = item["labels"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.tolist()
                # Use -100 for padding in labels (ignored by loss)
                padded_labels = labels + [-100] * (max_length - len(labels))
                padded_batch["labels"].append(padded_labels)
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            for key in padded_batch:
                padded_batch[key] = torch.tensor(padded_batch[key])
        
        return padded_batch

    
    def save_pretrained(self, save_directory):
        """Save tokenizer (copy the .model file)."""
        import shutil
        os.makedirs(save_directory, exist_ok=True)
        # Save a marker file
        with open(os.path.join(save_directory, "tokenizer_config.json"), 'w') as f:
            json.dump({"tokenizer_class": "SentencePieceTokenizerWrapper"}, f)


class CustomMarianTranslator:
    """
    MarianMT with custom SentencePiece tokenizer for Ibani.
    Best of both worlds: MarianMT speed + perfect character preservation.
    """
    
    def __init__(self,
                 base_model: str = "Helsinki-NLP/opus-mt-en-mul",
                 tokenizer_model: Optional[str] = None,
                 model_path: Optional[str] = None):
        """
        Initialize translator with custom tokenizer.
        
        Args:
            base_model: Base MarianMT model
            tokenizer_model: Path to custom SentencePiece model (.model file)
            model_path: Path to fine-tuned model (if available)
        """
        self.base_model = base_model
        self.tokenizer_model = tokenizer_model
        
        # Load model
        if model_path and os.path.exists(model_path):
            print(f"✓ Loading fine-tuned model from: {model_path}")
            self.model = MarianMTModel.from_pretrained(model_path)
            # Load custom tokenizer
            if tokenizer_model and os.path.exists(tokenizer_model):
                self.tokenizer = SentencePieceTokenizerWrapper(tokenizer_model)
            else:
                raise ValueError("Tokenizer model path required for fine-tuned model")
            model_source = "local"
        else:
            print(f"Loading base model: {base_model}")
            self.model = MarianMTModel.from_pretrained(base_model)
            
            # Use custom tokenizer if provided
            if tokenizer_model and os.path.exists(tokenizer_model):
                print(f"✓ Loading custom tokenizer: {tokenizer_model}")
                self.tokenizer = SentencePieceTokenizerWrapper(tokenizer_model)
                # Resize model embeddings to match new tokenizer
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"✓ Resized model embeddings to {len(self.tokenizer)}")
            else:
                raise ValueError("Custom tokenizer model path is required")
            
            model_source = "base"
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"✓ Using device: {self.device}")
        print(f"✓ Model source: {model_source}")
    
    def create_training_dataset(self, data_file: str) -> Dataset:
        """Create training dataset from JSON file."""
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training data file not found: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Dataset.from_list(data)
    
    def preprocess_data(self, examples):
        """Preprocess data for training."""
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["ibani"] for ex in examples["translation"]]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            text_target=targets,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def train_model(self,
                   training_data_file: str,
                   output_dir: str = "./ibani_custom_marian_model",
                   num_epochs: int = 5,
                   batch_size: int = 8,
                   learning_rate: float = 5e-5):
        """Train MarianMT with custom tokenizer."""
        print("Starting MarianMT training with custom tokenizer...")
        print(f"Training data: {training_data_file}")
        print(f"Output directory: {output_dir}")
        
        # Create training dataset
        dataset = self.create_training_dataset(training_data_file)
        print(f"Loaded {len(dataset)} training examples")
        
        # Preprocess
        print("Preprocessing data...")
        tokenized_dataset = dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            save_total_limit=2,
            predict_with_generate=True,
            logging_steps=50,
            save_steps=500,
            warmup_steps=100,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            report_to="none",
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        print("Training started...")
        print("=" * 70)
        trainer.train()
        
        # Save
        print("\nSaving model and tokenizer...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Also save the tokenizer model file
        import shutil
        if self.tokenizer_model:
            shutil.copy(self.tokenizer_model, os.path.join(output_dir, "tokenizer.model"))
        
        print(f"✓ Model training completed! Saved to {output_dir}")
    
    def translate(self, text: str, max_length: int = 128) -> str:
        """Translate English to Ibani."""
        if not text.strip():
            return ""
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation.strip()


def main():
    """Example usage."""
    print("="*70)
    print("MarianMT with Custom Tokenizer for Ibani")
    print("="*70)
    print("Combines MarianMT speed with perfect character preservation!")
    print("="*70)
    
    # Check if custom tokenizer exists
    tokenizer_path = "ibani_tokenizer.model"
    if not os.path.exists(tokenizer_path):
        print(f"\n❌ Custom tokenizer not found: {tokenizer_path}")
        print("Please run: python train_custom_tokenizer.py first")
        return
    
    # Initialize
    translator = CustomMarianTranslator(tokenizer_model=tokenizer_path)
    
    # Test
    test_sentences = [
        "I eat fish",
        "The woman goes",
        "Abraham was the father of Isaac"
    ]
    
    print("\nTesting with base model (before training):")
    for sentence in test_sentences:
        translation = translator.translate(sentence)
        print(f"EN: {sentence}")
        print(f"Translation: {translation}\n")


if __name__ == "__main__":
    main()
