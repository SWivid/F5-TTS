import argparse
import json
from pathlib import Path

import librosa
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="UTMOS Evaluation")
    parser.add_argument("--audio_dir", type=str, required=True, help="Audio file path.")
    parser.add_argument("--ext", type=str, default="wav", help="Audio extension.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device)

    audio_paths = list(Path(args.audio_dir).rglob(f"*.{args.ext}"))
    utmos_score = 0

    utmos_result_path = Path(args.audio_dir) / "_utmos_results.jsonl"
    with open(utmos_result_path, "w", encoding="utf-8") as f:
        for audio_path in tqdm(audio_paths, desc="Processing"):
            wav, sr = librosa.load(audio_path, sr=None, mono=True)
            wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)
            score = predictor(wav_tensor, sr)
            line = {}
            line["wav"], line["utmos"] = str(audio_path.stem), score.item()
            utmos_score += score.item()
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        avg_score = utmos_score / len(audio_paths) if len(audio_paths) > 0 else 0
        f.write(f"\nUTMOS: {avg_score:.4f}\n")

    print(f"UTMOS: {avg_score:.4f}")
    print(f"UTMOS results saved to {utmos_result_path}")


if __name__ == "__main__":
    main()
