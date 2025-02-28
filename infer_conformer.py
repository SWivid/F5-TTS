import torch
import soundfile as sf
import pandas as pd
import timeit
import json
# import IPython
# Installation /home/prasais/projects/xttsv2/NeMo
# Venv /home/prasais/projects/xttsv2/asr_venv

def get_conformer_transcripts():
    import nemo.collections.asr as nemo_asr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path='ai4b_indicConformer_ne.nemo')
    model.eval() # inference mode
    model = model.to(device) # transfer model to device

    def get_transcript(audio_path):
        ctc_text = model.transcribe([audio_path], batch_size=1,logprobs=False, language_id='ne')[0]
        return ctc_text[0]

    df = pd.read_csv("data/indic_r_char/metadata_orig.csv", delimiter = "|")
    base_path = "/home/prasais/projects/xttsv2/F5-TTS/data/indic_r_char/wavs_24000/"


    df['filepath'] = df['filepath'].apply(lambda x: base_path + x)
    df['filepath'] = df['filepath'].apply(lambda x: x.replace("_enhanced", "_enhanced_converted"))

    # Function to batch the list
    # test_df = df[:32]
    # df = df[:32]
    overall_start = timeit.default_timer()
    # test_df['transcript'] = test_df["filepath"].apply(get_transcript)
    # # Total time taken: 43.11 seconds.

    overall_end = timeit.default_timer()
    print(f"Total time taken: {overall_end - overall_start:.2f} seconds.")

    audio_files = df['filepath'].tolist()
    generation = {}
    for item in audio_files:
        transcriptions = model.transcribe([item], batch_size=1, language_id='ne', return_hypotheses=False, verbose=False)
        generation[item] = transcriptions[0]

    # transcriptions = model.transcribe([item], batch_size=32, num_workers=8, language_id='ne', return_hypotheses=False, verbose=False)
    with open("transcriptions.json", "w", encoding="utf-8") as f:
        json.dump(generation, f, ensure_ascii=False, indent=4)

def format_conformer_outputs():
    from jiwer import wer  # Word Error Rate (WER) from jiwer library


    with open("transcriptions.json", "r", encoding="utf-8") as f:
        transcriptions = json.load(f)

    original_df = pd.read_csv("data/indic_r_char/metadata_orig.csv", delimiter = "|")
    # translated_df = pd.DataFrame(list(transcriptions.items()), columns=['filepath', 'transcription'])
    translated_df = pd.DataFrame(
                        [(filepath, " ".join(transcription)) for filepath, transcription in transcriptions.items()], 
                        columns=['filepath', 'transcription']
                    )
    base_path = "/home/prasais/projects/xttsv2/F5-TTS/data/indic_r_char/wavs_24000/"
    translated_df['filepath'] = translated_df['filepath'].apply(lambda x: x.replace(base_path, ""))
    translated_df['filepath'] = translated_df['filepath'].apply(lambda x: x.replace("_enhanced_converted", "_enhanced"))
    
    merged_df = pd.merge(original_df, translated_df, on='filepath', how='inner')

    def calculate_wer(transcript, transcription):
        return wer(transcript, transcription)

    # Apply the function to each row in the DataFrame
    merged_df['wer'] = merged_df.apply(lambda row: calculate_wer(row['transcript'], row['transcription']), axis=1)
    # print(merged_df.head())
    filtered_df = merged_df[(merged_df['wer'] < 0.35) & (merged_df['transcript'].str.len() > 10) & (merged_df['transcript'].str.len() < 120)]

    # filtered_df = merged_df[merged_df['wer'] < 0.25]
    print(filtered_df.head())
    print(len(filtered_df))
    filtered_df = filtered_df.drop(["transcription", "wer"], axis=1)
    filtered_df.to_csv("metadata.csv", sep='|', index=False, header=False)


format_conformer_outputs()
