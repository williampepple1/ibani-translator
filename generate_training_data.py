#!/usr/bin/env python3
"""
Generate comprehensive training data from the Ibani dictionary.
This script creates training examples using the extensive ibani_dict.json file.
"""

import json
import random
from rule_based_translator import IbaniRuleBasedTranslator

def generate_training_data_from_dictionary():
    """Generate training data using the comprehensive Ibani dictionary."""
    
    # Load the comprehensive dictionary
    with open("ibani_dict.json", "r", encoding="utf-8") as f:
        dictionary_data = json.load(f)
    
    print(f"Loaded {len(dictionary_data)} dictionary entries")
    
    # Initialize rule-based translator
    translator = IbaniRuleBasedTranslator()
    
    # Create comprehensive training sentences
    training_sentences = []
    
    # Basic sentence patterns
    basic_patterns = [
        "I {verb} {noun}",
        "You {verb} {noun}", 
        "He {verb} {noun}",
        "She {verb} {noun}",
        "We {verb} {noun}",
        "They {verb} {noun}",
        "The {noun} {verb}",
        "The {noun} {verb} {noun}",
        "I {verb} you",
        "You {verb} me",
        "We {verb} them",
        "They {verb} us"
    ]
    
    # Get common verbs and nouns from dictionary
    verbs = []
    nouns = []
    
    for entry in dictionary_data:
        word = entry["word"].lower()
        pos = entry["Pos"]
        ibani_word = entry["Ibani_word"]
        
        if pos in ["v.", "v"] and len(word) > 2:
            verbs.append((word, ibani_word))
        elif pos in ["n.", "n"] and len(word) > 2:
            nouns.append((word, ibani_word))
    
    print(f"Found {len(verbs)} verbs and {len(nouns)} nouns in dictionary")
    
    # Generate sentences using patterns
    for pattern in basic_patterns:
        for _ in range(5):  # Generate 5 examples per pattern
            if "{verb}" in pattern and "{noun}" in pattern:
                if verbs and nouns:
                    verb_en, verb_ibani = random.choice(verbs)
                    noun_en, noun_ibani = random.choice(nouns)
                    sentence = pattern.format(verb=verb_en, noun=noun_en)
                    training_sentences.append(sentence)
            elif "{verb}" in pattern:
                if verbs:
                    verb_en, verb_ibani = random.choice(verbs)
                    sentence = pattern.format(verb=verb_en)
                    training_sentences.append(sentence)
            elif "{noun}" in pattern:
                if nouns:
                    noun_en, noun_ibani = random.choice(nouns)
                    sentence = pattern.format(noun=noun_en)
                    training_sentences.append(sentence)
    
    # Add tense variations
    tense_sentences = []
    for sentence in training_sentences[:20]:  # Take first 20 for tense variations
        tense_sentences.extend([
            sentence,  # Present
            sentence.replace("I ", "I will ").replace("You ", "You will ").replace("He ", "He will ").replace("She ", "She will ").replace("We ", "We will ").replace("They ", "They will "),  # Future
            sentence.replace("I ", "I have ").replace("You ", "You have ").replace("He ", "He has ").replace("She ", "She has ").replace("We ", "We have ").replace("They ", "They have "),  # Perfect
        ])
    
    training_sentences.extend(tense_sentences)
    
    # Remove duplicates
    training_sentences = list(set(training_sentences))
    
    print(f"Generated {len(training_sentences)} unique training sentences")
    
    # Translate all sentences using rule-based translator
    training_data = []
    for i, sentence in enumerate(training_sentences):
        try:
            ibani_translation = translator.translate_sentence(sentence)
            training_data.append({
                "translation": {
                    "en": sentence,
                    "ibani": ibani_translation
                }
            })
            
            if (i + 1) % 50 == 0:
                print(f"Translated {i + 1}/{len(training_sentences)} sentences")
                
        except Exception as e:
            print(f"Error translating '{sentence}': {e}")
            continue
    
    # Save training data
    with open("comprehensive_training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created comprehensive training data with {len(training_data)} examples")
    print("Saved to: comprehensive_training_data.json")
    
    return training_data

if __name__ == "__main__":
    generate_training_data_from_dictionary()
