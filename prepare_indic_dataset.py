import csv

unique_chars = set()

base_path = "data/indic_r_char/"

pre_vocab_file = "/home/prasais/projects/xttsv2/F5-TTS/ckpts/indic_r/vocab.txt"  # Update with your actual vocab file
with open(pre_vocab_file, "r", encoding="utf-8") as f:
    predefined_chars = set(f.read().splitlines())

# Initialize an empty set for extracted characters
extracted_chars = set()

# Open the input JSONL file and the output CSV file
with open(base_path+"metadata_train.json", "r", encoding="utf-8") as infile, open(base_path+"metadata.csv", "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile, delimiter="|")
    # writer.writerow(["file_name", "transcript"])  # Write header

    # Read each line as a separate JSON object
    for line in infile:
        try:
            entry = eval(line.strip())  # Convert string to dictionary
            file_name = entry.get("filepath", "")
            transcript = entry.get("verbatim", "")
            unique_chars.update(list(transcript))  # Splitting into characters
            writer.writerow([file_name, transcript])  # Write to CSV
        except Exception as e:
            print(f"Error processing line: {line[:50]}... - {e}")


# Find missing characters
missing_chars = extracted_chars - predefined_chars
# Update the character set
updated_chars = predefined_chars.union(missing_chars)

# Save the updated vocab
output_vocab_file = base_path+"vocab.txt"
with open(output_vocab_file, "w", encoding="utf-8") as f:
    for char in sorted(updated_chars):
        f.write(char + "\n")

print("Conversion completed! Files saved: metadata.csv & vocab.txt")
print("Conversion completed! Output saved in metadata.csv")
