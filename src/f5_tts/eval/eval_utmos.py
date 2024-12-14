import torch
import librosa
from pathlib import Path
import json
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate UTMOS scores for audio files.")
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Path to the directory containing WAV audio files."
    )
    parser.add_argument("--ext", type=str, default="wav", help="audio extension.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g. 'cuda' or 'cpu').")

    args = parser.parse_args()

    device = "cuda" if args.device and torch.cuda.is_available() else "cpu"

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device)

    lines = list(Path(args.audio_dir).rglob(f"*.{args.ext}"))
    results = {}
    utmos_result = 0

    for line in tqdm(lines, desc="Processing"):
        wave_name = line.stem
        wave, sr = librosa.load(line, sr=None, mono=True)
        wave_tensor = torch.from_numpy(wave).to(device).unsqueeze(0)
        score = predictor(wave_tensor, sr)
        results[str(wave_name)] = score.item()
        utmos_result += score.item()

    avg_score = utmos_result / len(lines) if len(lines) > 0 else 0
    print(f"UTMOS: {avg_score}")

    output_path = Path(args.audio_dir) / "utmos_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results have been saved to {output_path}")


if __name__ == "__main__":
    main()
