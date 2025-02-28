#pip install ai4bharat-transliteration
from ai4bharat.transliteration import XlitEngine
import torch
import argparse
import pandas as pd
import re
import json
from collections import defaultdict

torch.serialization.add_safe_globals([argparse.Namespace])


engine = XlitEngine("ne", beam_width=4, rescore=True, src_script_type = "en")



def translit_nepali_to_english():
    nepali_text = "सूर्यविनायक नगरपालिका–९ नलिञ्चोकमा गए राति विपरीत दिशाबाट आएका तीन मोटरसाइकल एकापसमा ठोक्किँदा एक जनाको मृत्यु भएको छ भने अन्य दुई जना घाइते भएका छन् ।"

    engine = XlitEngine(src_script_type="indic", beam_width=10, rescore=False)
    result = engine.translit_sentence(nepali_text, lang_code="ne")
    return result


def translit_word_english_to_nepali(text):
    # result = engine.translit_sentence(english_text, lang_code="ne")
    # transliterate word 
    # out = e.translit_word("one", topk=1)
    result = engine.translit_word(text, lang_code="ne", topk=1)
    return result


def clean_and_split(text):
    # Remove non-alphabetic characters (except spaces), convert to lowercase, and split into words
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    return cleaned_text.split()


df = pd.read_csv("data/libritts_100_char/metadata.csv", delimiter="|")

# Apply the function to the 'transcript' column and get unique words
# '''
word_count_dict = defaultdict(int)

unique_words = set()
for transcript in df['transcript']:
    unique_words.update(clean_and_split(transcript))

with open('unique_words.json', 'w') as json_file:
    json.dump(list(unique_words), json_file, indent=4)

'''
unique_words = {}
for transcript in df['transcript']:
    for word in clean_and_split(transcript):
        # Capitalize the first letter and store as key-value pair in the dictionary
        translit_word = translit_word_english_to_nepali(word)
        unique_words[word] = translit_word

with open('word_dict.json', 'w') as json_file:
    json.dump(word_dict, json_file, indent=4)
'''
"""

unique_words = json.load('word_dict.json')
for transcript in df['transcript']:
    for word in clean_and_split(transcript):
        word_count_dict[word] += 1

# Sort words by length (longest to shortest)
sorted_words = sorted(word_dict.keys(), key=len, reverse=True)

# Function to replace words in a transcript
def replace_words_in_transcript(text):
    for word in sorted_words:
        if word in text:
            text = text.replace(word, word_dict[word])
    return text

# Replace words in the 'transcript' column with the transformed ones
df['transcript'] = df['transcript'].apply(replace_words_in_transcript)

# Save the updated DataFrame to a new CSV file
df.to_csv("updated_metadata.csv", index=False)



"""