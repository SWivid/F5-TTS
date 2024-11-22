from torch.utils.data import Dataset
from .text_processor import MongolianTextProcessor
from .vocabulary import Vocabulary
import torchaudio
import torch.nn.functional as F
import torch

class MongolianAwareDataset(Dataset):
    def __init__(self, dataset, target_sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256, n_mel_channels=100, vocab_tokens=None):
        self.dataset = dataset
        self.text_processor = MongolianTextProcessor()
        self.vocab = Vocabulary(specials=list(self.text_processor.special_tokens) + ['<pad>', '<unk>'])
        
        if vocab_tokens:
            self.vocab = Vocabulary(specials=vocab_tokens)
        else:
            self.vocab = Vocabulary(specials=list(self.text_processor.special_tokens) + ['<pad>', '<unk>'])
            self.build_vocab()
        
        # Store parameters as instance variables
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mel_channels = n_mel_channels
        
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mel_channels,
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def build_vocab(self):
        # Iterate over the dataset to build the vocabulary
        for item in self.dataset:
            tokens_with_position, _, _ = self.text_processor.process_text(item['text'])
            for token, _ in tokens_with_position:
                self.vocab.add_token(token)
    
    def __getitem__(self, index):
        item = self.dataset[index]

        # Load audio file
        audio_path = item['audio_path']
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        # Convert waveform to mel spectrogram
        mel_spectrogram = self.mel_spectrogram_transform(waveform).squeeze(0)  # Shape: [n_mels, time]

        # Process text
        tokens_with_position, weights, prosody_features = self.text_processor.process_text(item['text'])

        # Ensure prosody features are properly sized
        prosody_features = {k: v[:mel_spectrogram.shape[1]] for k, v in prosody_features.items()}

        # Update item with processed information
        item.update({
            'tokens_with_position': tokens_with_position,
            'phoneme_weights': weights,
            'prosody_features': prosody_features,
            'mel': mel_spectrogram,
            'mel_lengths': mel_spectrogram.shape[1],
            'original_text': item['text'],
            'vocab': self.vocab
        })

        return item



def collate_fn(batch):
    # Check if 'vocab' exists in the batch
    if 'vocab' in batch[0]:
        vocab = batch[0]['vocab']
    else:
        raise KeyError("'vocab' key not found in the dataset items. Ensure the dataset includes 'vocab'.")

    # Prepare mel spectrograms and lengths
    mels = [item['mel'] for item in batch]
    mel_lengths = [mel.shape[1] for mel in mels]

    # Pad mel spectrograms to the max length
    max_mel_length = max(mel_lengths)
    padded_mels = [F.pad(mel, (0, max_mel_length - mel.shape[1])) for mel in mels]
    mels_tensor = torch.stack(padded_mels)

    # Prepare prosody features
    max_token_len = max_mel_length  # Match with the mel spectrogram length
    prosody_features_list = {'emphasis': [], 'pause_duration': [], 'intonation': []}

    for item in batch:
        for key in prosody_features_list.keys():
            curr_feature = item['prosody_features'][key]
            # Pad prosody features to match mel spectrogram length
            padded_feature = F.pad(curr_feature, (0, max_token_len - curr_feature.shape[0]))
            prosody_features_list[key].append(padded_feature)

    # Stack tensors for prosody features
    for key in prosody_features_list.keys():
        prosody_features_list[key] = torch.stack(prosody_features_list[key])  # Shape: [batch_size, max_mel_length]

    # Convert tokens to tensor
    tokens = []
    for item in batch:
        curr_tokens = [vocab[token] for token, _ in item['tokens_with_position']]  # Map tokens to vocab indices
        tokens.append(curr_tokens)

    # Pad tokens to max length
    max_token_len = max(len(t) for t in tokens)
    padded_tokens = [t + [vocab['<pad>']] * (max_token_len - len(t)) for t in tokens]
    tokens_tensor = torch.tensor(padded_tokens, dtype=torch.long)  # Shape: [batch_size, max_token_len]

    # Prepare batch dictionary
    batch = {
        'mel': mels_tensor,  # Shape: [batch_size, n_mels, max_mel_length]
        'mel_lengths': torch.tensor(mel_lengths, dtype=torch.long),
        'prosody_features': prosody_features_list,
        'text': tokens_tensor,  # Token indices
        'original_text': [item['original_text'] for item in batch],
        'vocab': vocab
    }

    return batch


