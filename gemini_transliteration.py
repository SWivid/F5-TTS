#gemini transliteration

import json
import glob

# with open('unique_words.json', 'r') as json_file:
#     loaded_list = json.load(json_file)

# unique_words= set(loaded_list)


translit_words = {}

phonetic_files_path = "/home/prasais/projects/xttsv2/F5-TTS/gemini_transliterations/"
phonetic_files = glob.glob(phonetic_files_path + "*.json")
# print("Reading from files: ", phonetic_files)
# Loop through each phonetic JSON file
for file in phonetic_files:
    with open(file, 'r') as json_file:
        data = json.load(json_file)
    translit_words.update({item['original_word']: item['phonetic_nepali'] for item in data})

44# print(translit_words)
with open('english_nepali_transliteration_gemini.json', 'w') as json_file:
    json.dump(translit_words, json_file, indent=4)
