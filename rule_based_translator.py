"""
Rule-based English to Ibani translator with grammar rules.
This serves as a baseline before implementing ML-based translation.
"""

import json
import re
import os
from typing import List, Dict, Optional


class IbaniRuleBasedTranslator:
    def __init__(self, dictionary_path: str = "ibani_dict.json"):
        """Initialize the translator with Ibani dictionary."""
        try:
            with open(dictionary_path, "r", encoding="utf-8") as f:
                self.dictionary_data = json.load(f)
        except FileNotFoundError:
            print(f"Dictionary file not found at: {dictionary_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            raise
        
        # Convert the new format to a lookup dictionary
        self.dictionary = {}
        self.pos_info = {}  # Store part of speech information
        
        for entry in self.dictionary_data:
            english_word = entry["word"].lower().strip()
            ibani_word = entry["Ibani_word"]
            pos = entry["Pos"]
            
            # Handle multiple English words (comma-separated)
            if "," in english_word:
                words = [w.strip() for w in english_word.split(",")]
                for word in words:
                    self.dictionary[word] = ibani_word
                    self.pos_info[word] = pos
            else:
                self.dictionary[english_word] = ibani_word
                self.pos_info[english_word] = pos
        
        # Add common verb mappings for better translation
        common_verbs = {
            "eat": "fii",
            "go": "mu",
            "come": "bo",
            "see": "ari",
            "hear": "na",
            "speak": "egere",
            "run": "mangi",
            "sit": "kporofini",
            "sleep": "muno",
            "wake": "saghi"
        }
        
        for eng_verb, ibani_verb in common_verbs.items():
            if eng_verb not in self.dictionary:
                self.dictionary[eng_verb] = ibani_verb
                self.pos_info[eng_verb] = "v."
        
        # Grammar rules for Ibani
        self.grammar_rules = {
            "word_order": "SOV",  # Subject-Object-Verb
            "negation": "suffix",  # Negation comes after verb
            "tense_markers": {
                "present": "",
                "past": "m",  # Add 'm' suffix for past tense
                "future": "bem",  # Add 'bem' suffix for future tense
                "perfect": "mam",  # Add 'mam' suffix for perfect aspect (have + verb)
                "progressive": "ari",  # Add 'ari' suffix for progressive aspect (verb + ing)
                "negative_future": "bigha"  # Add 'bigha' suffix for negative future (will not + verb)
            }
        }
    
    def translate_word(self, word: str) -> str:
        """Translate a single English word to Ibani."""
        # Clean the word (remove punctuation, convert to lowercase)
        # Clean the word (remove punctuation, keep letters and tonal marks)
        # We use a regex that keeps alphanumeric characters and common combining marks
        clean_word = re.sub(r'[^\w\u0300-\u036f\u0323]', '', word.lower())
        
        # Look up in dictionary
        translation = self.dictionary.get(clean_word, word)
        
        # If not found, try to handle common English suffixes
        if translation == word and len(clean_word) > 3:
            # Try without common suffixes
            for suffix in ['ing', 'ed', 'er', 'est', 'ly', 's']:
                if clean_word.endswith(suffix):
                    base_word = clean_word[:-len(suffix)]
                    base_translation = self.dictionary.get(base_word)
                    if base_translation:
                        return base_translation
        
        return translation
    
    def detect_tense_and_aspect(self, words: List[str]) -> Dict[str, str]:
        """Detect tense and aspect from English words."""
        tense_info = {
            "tense": "present",
            "aspect": "simple",
            "negation": False
        }
        
        word_text = " ".join(words).lower()
        
        # Check for negation first
        if any(indicator in word_text for indicator in ["not", "n't", "no", "never"]):
            tense_info["negation"] = True
        
        # Check for perfect aspect (have/has + verb) - this takes precedence
        if any(indicator in word_text for indicator in ["have", "has", "had"]):
            tense_info["aspect"] = "perfect"
            # Determine tense for perfect aspect
            if "had" in word_text:
                tense_info["tense"] = "past"
            else:
                tense_info["tense"] = "present"
        
        # Check for progressive aspect (be + verb + ing)
        elif any(indicator in word_text for indicator in ["am", "is", "are", "was", "were"]) and "ing" in word_text:
            tense_info["aspect"] = "progressive"
            # Determine tense for progressive aspect
            if any(indicator in word_text for indicator in ["was", "were"]):
                tense_info["tense"] = "past"
            else:
                tense_info["tense"] = "present"
        
        # Check for simple past tense (regular verbs with -ed or irregular past forms)
        elif any(word.endswith("ed") for word in words) or any(word in ["ate", "went", "came", "saw", "ran", "drank", "slept", "woke"] for word in words):
            tense_info["tense"] = "past"
            tense_info["aspect"] = "simple"
        
        # Check for past tense with auxiliary verbs
        elif any(indicator in word_text for indicator in ["was", "were", "did"]):
            tense_info["tense"] = "past"
            tense_info["aspect"] = "simple"
        
        # Check for future tense indicators
        elif any(indicator in word_text for indicator in ["will", "shall", "going to"]):
            tense_info["tense"] = "future"
            tense_info["aspect"] = "simple"
        
        # Default: simple present (no special markers)
        else:
            tense_info["tense"] = "present"
            tense_info["aspect"] = "simple"
        
        return tense_info
    
    def apply_verb_morphology(self, verb: str, tense_info: Dict[str, str]) -> str:
        """Apply Ibani verb morphology based on tense and aspect."""
        if not verb:
            return verb
        
        # Get the base verb from dictionary or use common mappings
        base_verb = self.dictionary.get(verb.lower(), verb.lower())
        
        # If not found in dictionary, try to get base form
        if base_verb == verb.lower():
            # Handle common English verb forms
            verb_forms = {
                "ate": "eat", "eaten": "eat", "eating": "eat",
                "went": "go", "gone": "go", "going": "go", 
                "came": "come", "coming": "come",
                "drank": "drink", "drunk": "drink", "drinking": "drink",
                "saw": "see", "seen": "see", "seeing": "see",
                "ran": "run", "running": "run"
            }
            base_english = verb_forms.get(verb.lower(), verb.lower())
            base_verb = self.dictionary.get(base_english, base_english)
        
        # Apply tense and aspect markers
        if tense_info["aspect"] == "perfect":
            # Perfect aspect: add 'mam' suffix
            return base_verb + "mam"
        elif tense_info["aspect"] == "progressive":
            # Progressive aspect: add 'ari' suffix
            return base_verb + "ari"
        elif tense_info["tense"] == "past":
            # Past tense: add 'm' suffix
            return base_verb + "m"
        elif tense_info["tense"] == "future":
            if tense_info["negation"]:
                # Negative future: add 'bigha' suffix
                return base_verb + "bigha"
            else:
                # Future tense: add 'bem' suffix
                return base_verb + "bem"
        
        return base_verb
    
    def identify_sentence_structure(self, words: List[str]) -> Dict[str, List[str]]:
        """Identify parts of speech in the sentence using dictionary POS info."""
        structure = {
            "subject": [],
            "verb": [],
            "object": [],
            "modifiers": []
        }
        
        # Simple approach: first word is subject, last meaningful word is verb, middle words are objects
        if len(words) >= 2:
            # Handle "The X" pattern - subject is the word after "the"
            if words[0].lower() == "the" and len(words) > 1:
                structure["subject"].append(words[1])  # Subject is the word after "the"
            else:
                structure["subject"].append(words[0])  # First word is subject
            
            # Find the main verb (usually the last meaningful word or a known verb)
            main_verb = None
            for i, word in enumerate(words):
                clean_word = word.lower()
                if (clean_word in ["eat", "drink", "go", "come", "see", "hear", "speak", "walk", "run", "ate", "went", "eaten", "gone", "eating", "going"] or
                    self.pos_info.get(clean_word, "") in ["v.", "v"]):
                    main_verb = word
                    break
            
            # If no verb found, use the last word
            if not main_verb and len(words) > 1:
                main_verb = words[-1]
            
            if main_verb:
                structure["verb"].append(main_verb)
            
            # Everything else is object or modifiers (skip words already in subject)
            for word in words[1:]:
                if (word != main_verb and 
                    word.lower() != "the" and 
                    word not in structure["subject"]):
                    structure["object"].append(word)
        
        return structure
    
    def apply_grammar_rules(self, words: List[str], tense: str = "present") -> List[str]:
        """Apply Ibani grammar rules to rearrange words and apply verb morphology."""
        if len(words) < 2:
            return words
        
        # Keep original English words for structure analysis
        original_words = words.copy()
        
        # Detect tense and aspect from the original English words
        tense_info = self.detect_tense_and_aspect(original_words)
        
        # Identify sentence structure using original English words
        structure = self.identify_sentence_structure(original_words)
        
        # Apply SOV (Subject-Object-Verb) word order
        if self.grammar_rules["word_order"] == "SOV":
            result = []
            
            # Add subject (filter out auxiliary verbs and negation)
            subject_words = []
            for word in structure["subject"]:
                if word.lower() not in ["have", "has", "had", "will", "shall", "am", "is", "are", "was", "were", "not", "n't", "the"]:
                    # Translate the subject word
                    translated_subject = self.translate_word(word)
                    subject_words.append(translated_subject)
            
            # Add "má" (the) after the subject if "the" was in the original
            if "the" in [w.lower() for w in words]:
                # Put "má" after the translated subject (woman má, not má woman)
                subject_words.append("má")
            
            result.extend(subject_words)
            
            # Add object (filter out auxiliary verbs and negation)
            for word in structure["object"]:
                if word.lower() not in ["have", "has", "had", "will", "shall", "am", "is", "are", "was", "were", "not", "n't", "the"]:
                    # Translate object words
                    translated_object = self.translate_word(word)
                    result.append(translated_object)
            
            # Add modifiers (filter out auxiliary verbs and negation)
            for word in structure["modifiers"]:
                if word.lower() not in ["have", "has", "had", "will", "shall", "am", "is", "are", "was", "were", "not", "n't", "the"]:
                    # Translate modifier words
                    translated_modifier = self.translate_word(word)
                    result.append(translated_modifier)
            
            # Add verb with proper morphology
            verb = structure["verb"]
            if verb:
                # Apply verb morphology based on tense and aspect
                for i, verb_word in enumerate(verb):
                    # Translate the verb first
                    translated_verb = self.translate_word(verb_word)
                    # Apply morphology
                    morphed_verb = self.apply_verb_morphology(translated_verb, tense_info)
                    verb[i] = morphed_verb
                
                result.extend(verb)
            
            return result
        
        return words
    
    def translate_sentence(self, sentence: str, tense: str = "present") -> str:
        """Translate an English sentence to Ibani."""
        # Tokenize the sentence
        # Tokenize the sentence, keeping words with their tonal marks
        words = re.findall(r"[\w\u0300-\u036f\u0323]+", sentence)
        
        if not words:
            return ""
        
        # Apply grammar rules first using original English words
        final_words = self.apply_grammar_rules(words, tense)
        
        # Join into Ibani sentence
        return " ".join(final_words)
    
    def translate_text(self, text: str) -> str:
        """Translate a longer text (multiple sentences)."""
        sentences = re.split(r'[.!?]+', text)
        translated_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                translated = self.translate_sentence(sentence)
                translated_sentences.append(translated)
        
        return ". ".join(translated_sentences)


def main():
    """Example usage of the rule-based translator."""
    translator = IbaniRuleBasedTranslator()
    
    # Test sentences demonstrating different tenses and aspects
    test_sentences = [
        "I eat fish",  # Present
        "I ate fish",  # Past
        "I will eat fish",  # Future
        "I have eaten fish",  # Perfect
        "I am eating fish",  # Progressive
        "I will not eat fish",  # Negative future
        "The woman goes",  # Present
        "The woman went",  # Past
        "The woman will go",  # Future
        "The woman has gone",  # Perfect
        "The woman is going",  # Progressive
        "The woman will not go",  # Negative future
        "The man slapped me",  # SOV example
        "The child sees the woman",  # SOV with object
        "I see you",  # Simple SOV
        "The woman eats fish"  # SOV with object
    ]
    
    print("Rule-based English to Ibani Translation:")
    print("=" * 50)
    
    for sentence in test_sentences:
        translation = translator.translate_sentence(sentence)
        print(f"English: {sentence}")
        print(f"Ibani:   {translation}")
        print()


if __name__ == "__main__":
    main()
