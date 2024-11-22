import os
import torch
from torch.utils.data import DataLoader

from f5_tts.mongolian.model import ProsodyAwareDiT
from f5_tts.mongolian.dataset import MongolianAwareDataset, collate_fn
from f5_tts.model import CFM
from f5_tts.train.train_mongolian import setup_model, load_checkpoint


# Function to load dataset
def load_dataset(dataset_path):
    data = []
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            filename, text = line.strip().split('|')
            audio_path = os.path.join(dataset_path, 'wavs', filename + '.wav')
            data.append({'filename': filename, 'text': text, 'audio_path': audio_path})
    return data


# Validation function
def validate(model, validation_loader, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch in validation_loader:
            # Move batch data to device
            mel = batch['mel'].to(device)
            text = batch['text'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            prosody_features = {
                'emphasis': batch['prosody_features']['emphasis'].to(device),
                'pause_duration': batch['prosody_features']['pause_duration'].to(device),
                'intonation': batch['prosody_features']['intonation'].to(device)
            }

            # Forward pass
            loss, _, _ = model(
                inp=mel.permute(0, 2, 1),  # Adjust shape if needed
                text=text,
                lens=mel_lengths,
                prosody_features=prosody_features
            )

            total_loss += loss.item() * mel.size(0)  # Accumulate loss scaled by batch size

    # Calculate average loss
    avg_loss = total_loss / len(validation_loader.dataset)
    return avg_loss

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f]
    return tokens

# Main function
def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = os.path.join(args.checkpoint_dir, "vocab.txt")
    vocab_tokens = load_vocab(vocab_path)
    
    # Load validation dataset
    validation_data = load_dataset(args.validation_dataset_path)
    validation_dataset = MongolianAwareDataset(dataset=validation_data, vocab_tokens=vocab_tokens)
    validation_dataloader = DataLoader(
        validation_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Load model
    vocab_size = len(validation_dataset.vocab)
    model_dim = 1024  # Match the model_dim used in training
    model_depth = 22  # Match the model_depth used in training
    model = setup_model(vocab_size, model_dim=model_dim, model_depth=model_depth, n_mel_channels=100).to(device)

    # Load checkpoint
    checkpoint_path = args.checkpoint_path
    load_checkpoint(checkpoint_path, model)

    # Define loss function
    loss_fn = torch.nn.MSELoss()

    # Validate model
    validation_loss = validate(model, validation_dataloader, loss_fn, device)
    print(f"Validation Loss: {validation_loss}")




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a TTS model checkpoint.")
    parser.add_argument("--validation_dataset_path", required=True, help="Path to the validation dataset.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for validation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    args = parser.parse_args()

    main(args)
