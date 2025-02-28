import pandas as pd
import json
import re

df = pd.read_csv("data/libritts_100_char/metadata.csv", delimiter="|")

with open("unique_words.json", 'r') as json_file:
    unique_word_dict = json.load(json_file)

with open("english_nepali_transliteration_gemini.json", 'r') as json_file:
    word_dict = json.load(json_file)

sorted_words = sorted(word_dict.keys(), key=len, reverse=True)

def clean_and_split(text):
    # Remove non-alphabetic characters (except spaces), convert to lowercase, and split into words
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    return cleaned_text

# Function to replace words in a transcript
def replace_words_in_transcript(text):
    text = clean_and_split(text)
    # print(text)
    # quit()
    for word in sorted_words:
        pattern = r'\b' + re.escape(word) + r'\b'  # Match whole word with word boundaries
        text = re.sub(pattern, word_dict[word], text)
    return text

# Run this
# df['transcript'] = df['transcript'].apply(replace_words_in_transcript)
# df.to_csv("updated_metadata.csv",delimiter="|", index=False)

remaining = set(unique_word_dict) - set(word_dict.keys())
with open('remaining_words.json', 'w') as json_file:
    json.dump(list(remaining), json_file, indent=4)


