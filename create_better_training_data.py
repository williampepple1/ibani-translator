#!/usr/bin/env python3
"""
Create better training data using natural sentence patterns with the Ibani dictionary.
"""

import json
from rule_based_translator import IbaniRuleBasedTranslator

def create_natural_training_data():
    """Create natural training sentences using the Ibani dictionary."""
    
    # Natural sentence patterns that work well for translation
    natural_sentences = [
        # Basic present tense
        "I eat fish",
        "You drink water", 
        "He sees the woman",
        "She goes home",
        "We hear music",
        "They speak English",
        "The man walks",
        "The woman sits",
        "The child runs",
        "The dog barks",
        
        # Past tense
        "I ate fish",
        "You drank water",
        "He saw the woman", 
        "She went home",
        "We heard music",
        "They spoke English",
        "The man walked",
        "The woman sat",
        "The child ran",
        "The dog barked",
        
        # Future tense
        "I will eat fish",
        "You will drink water",
        "He will see the woman",
        "She will go home", 
        "We will hear music",
        "They will speak English",
        "The man will walk",
        "The woman will sit",
        "The child will run",
        "The dog will bark",
        
        # Perfect aspect
        "I have eaten fish",
        "You have drunk water",
        "He has seen the woman",
        "She has gone home",
        "We have heard music",
        "They have spoken English",
        "The man has walked",
        "The woman has sat",
        "The child has run",
        "The dog has barked",
        
        # Progressive aspect
        "I am eating fish",
        "You are drinking water",
        "He is seeing the woman",
        "She is going home",
        "We are hearing music",
        "They are speaking English",
        "The man is walking",
        "The woman is sitting",
        "The child is running",
        "The dog is barking",
        
        # SOV examples
        "The man slapped me",
        "The woman hit him",
        "The child pushed her",
        "The dog bit me",
        "The cat scratched you",
        "The bird pecked him",
        "The fish swam away",
        "The tree grew tall",
        "The flower bloomed",
        "The sun shone bright",
        
        # More complex sentences
        "The man sees the woman",
        "The woman hears the man",
        "The child follows the dog",
        "The dog chases the cat",
        "The cat catches the mouse",
        "The bird flies high",
        "The fish swims deep",
        "The tree stands tall",
        "The flower smells sweet",
        "The sun shines warm",
        
        # Questions and statements
        "What is your name?",
        "How are you?",
        "Where are you going?",
        "When will you come?",
        "Why did you go?",
        "Who is that man?",
        "Which way is home?",
        "How much does it cost?",
        "What time is it?",
        "Where is the house?",
        
        # Common phrases
        "Good morning",
        "Good evening", 
        "Thank you",
        "You're welcome",
        "I'm sorry",
        "Excuse me",
        "Please help me",
        "Can you help?",
        "I need water",
        "I want food",
        "I love you",
        "I miss you",
        "I'm happy",
        "I'm sad",
        "I'm tired",
        "I'm hungry",
        "I'm thirsty",
        "I'm cold",
        "I'm hot",
        "I'm fine"
    ]
    
    # Initialize translator
    translator = IbaniRuleBasedTranslator()
    
    # Translate all sentences
    training_data = []
    for i, sentence in enumerate(natural_sentences):
        try:
            ibani_translation = translator.translate_sentence(sentence)
            training_data.append({
                "translation": {
                    "en": sentence,
                    "ibani": ibani_translation
                }
            })
            print(f"Translated: '{sentence}' → '{ibani_translation}'")
            
        except Exception as e:
            print(f"Error translating '{sentence}': {e}")
            continue
    
    # Save training data
    with open("natural_training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nCreated natural training data with {len(training_data)} examples")
    print("Saved to: natural_training_data.json")
    
    return training_data

if __name__ == "__main__":
    create_natural_training_data()
