"""
Train MarianMT with custom SentencePiece tokenizer for Ibani.
Uses T5Tokenizer as a wrapper - fully compatible with HuggingFace!
"""

import json
import torch
from transformers import (
    MarianMTModel,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from typing import Optional
import os
import shutil


class CustomMarianTranslator:
    """
    MarianMT with custom SentencePiece tokenizer for Ibani.
    Uses T5Tokenizer wrapper for full HuggingFace compatibility.
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
            # Load tokenizer from saved model
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            model_source = "local"
        else:
            print(f"Loading base model: {base_model}")
            self.model = MarianMTModel.from_pretrained(base_model)
            
            # Use T5Tokenizer with custom SentencePiece model
            if tokenizer_model and os.path.exists(tokenizer_model):
                print(f"✓ Loading custom tokenizer: {tokenizer_model}")
                
                # Create temporary directory for T5Tokenizer
                temp_dir = "temp_tokenizer"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Copy SentencePiece model
                shutil.copy(tokenizer_model, os.path.join(temp_dir, "spiece.model"))
                
                # Create minimal tokenizer config
                tokenizer_config = {
                    "model_max_length": 512,
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                    "extra_ids": 0
                }
                
                with open(os.path.join(temp_dir, "tokenizer_config.json"), 'w') as f:
                    json.dump(tokenizer_config, f)
                
                # Load T5Tokenizer
                self.tokenizer = T5Tokenizer.from_pretrained(temp_dir, legacy=False)
                
                # Clean up temp directory
                shutil.rmtree(temp_dir)
                
                # Resize model embeddings
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
        
        # Replace padding token id with -100 so it's ignored by loss
        labels_ids = labels["input_ids"]
        labels_ids = [
            [(label if label != self.tokenizer.pad_token_id else -100) for label in label_seq]
            for label_seq in labels_ids
        ]
        
        model_inputs["labels"] = labels_ids
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
            processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
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
        
        # Also save the original tokenizer model file
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
    print("Using T5Tokenizer wrapper - fully compatible!")
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
