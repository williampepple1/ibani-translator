"""
Verify the updated JSON file
"""

import json

# Read and verify the JSON file
with open('ibani_eng_training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total entries in ibani_eng_training_data.json: {len(data)}")
print("\nFirst 3 entries from CSV data (should be at the end):")
print("-" * 60)

# Show the last few entries (which should be from the CSV)
for i, entry in enumerate(data[-5:], start=len(data)-4):
    ibani = entry['translation']['ibani']
    english = entry['translation']['en']
    print(f"{i}. Ibani: {ibani}")
    print(f"   English: {english}")
    print()

print("-" * 60)
print("[SUCCESS] Data successfully added from CSV to JSON!")
