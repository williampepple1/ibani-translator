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
from datasets import Dataset
from typing import List, Dict, Optional
import os
from pathlib import Path


class IbaniHuggingFaceTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-mul", model_path: Optional[str] = None):
        """
        Initialize the Hugging Face translator.
        
        Args:
            model_name: Pre-trained model to use as base (English-Swahili as starting point)
            model_path: Path to fine-tuned model (if available)
        """
        self.model_name = model_name
        self.model_path = model_path
        
        # Load tokenizer and model
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_path)
            self.model = MarianMTModel.from_pretrained(model_path)
        else:
            print(f"Loading pre-trained model: {model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
    
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
            
        except Exception as e:
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
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["ibani"] for ex in examples["translation"]]
        
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
        print("🚀 Starting model training...")
        
        # Create training dataset
        dataset = self.create_training_dataset(training_data_file)
        
        # Preprocess the data
        tokenized_dataset = dataset.map(
            self.preprocess_data, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split into train and validation
        train_size = int(0.8 * len(tokenized_dataset))
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="no",  # Disable evaluation for simplicity
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
        
        print(f"✅ Model training completed! Model saved to {output_dir}")
        
        # Update model path for future use
        self.model_path = output_dir
    
    def translate(self, text: str) -> str:
        """Translate English text to Ibani."""
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
                early_stopping=True
            )
        
        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
    print("🤖 Hugging Face English to Ibani Translator")
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
