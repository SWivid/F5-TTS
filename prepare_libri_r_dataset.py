import os
import shutil
import csv
from pathlib import Path

# Run below command in test-clean folder of librispeech. It moves all files into main directory.
# Can run this code after that
# find . -maxdepth 3 -type f -exec mv -t . {} + && find . -mindepth 1 -type d -exec rm -r {} +

def convert_libritts_to_standard_format(input_dir, output_dir):
    # Create directory structure
    wavs_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)
    
    metadata = []
    
    # Process all files in input directory
    for file in Path(input_dir).glob('*.wav'):
        # Get corresponding normalized text file
        txt_file = file.with_suffix('.normalized.txt')
        
        if not txt_file.exists():
            continue  # Skip files without normalized text
            
        # Read transcript
        with open(txt_file, 'r') as f:
            transcript = f.read().strip()
            
        # Copy audio file to wavs directory
        new_audio_path = os.path.join(wavs_dir, file.name)
        shutil.copy(file, new_audio_path)
        
        # Add to metadata
        metadata.append({
            'file_name': file.name,
            'transcript': transcript
        })
    
    # Write metadata.csv
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file_name', 'transcript'], delimiter='|')
        writer.writeheader()
        writer.writerows(metadata)

if __name__ == '__main__':
    input_directory = '/home/prasais/projects/xttsv2/libri_r_subset/LibriTTS/train-clean-100'
    output_directory = '/home/prasais/projects/xttsv2/F5-TTS/data/LibriTTS_100_char/dataset'
    
    convert_libritts_to_standard_format(input_directory, output_directory)
    print(f"Dataset converted and saved to {output_directory}")
