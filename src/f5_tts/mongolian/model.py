import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from f5_tts.model.modules import (
    TimestepEmbedding,
    AdaLayerNormZero,
    AdaLayerNormZero_Final,
    Attention,
    AttnProcessor,
    FeedForward
)

from f5_tts.model.backbones.dit import (
    TextEmbedding,
    InputEmbedding
)

# Constants
DEFAULT_DIM = 1024
MAX_SEQ_LEN = 4096

class ProsodyEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Embeddings for different prosody features
        self.emphasis_embedding = nn.Linear(1, dim)
        self.pause_embedding = nn.Linear(1, dim)
        self.intonation_embedding = nn.Linear(3, dim)
        
        # Position embedding
        self.pos_embedding = nn.Embedding(4096, dim)  # Max sequence length of 4096
        
        # Integration layers
        self.prosody_mixer = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        # Multi-head attention
        self.prosody_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, prosody_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            prosody_features: Dict containing:
                - emphasis: [batch_size, seq_len]
                - pause_duration: [batch_size, seq_len]
                - intonation: [batch_size, seq_len]
        Returns:
            prosody_encoding: [batch_size, seq_len, dim]
        """
        batch_size, seq_len = prosody_features['emphasis'].shape
        
        # Embed each feature [batch_size, seq_len, 1] -> [batch_size, seq_len, dim]
        emphasis = self.emphasis_embedding(prosody_features['emphasis'].unsqueeze(-1))
        pause = self.pause_embedding(prosody_features['pause_duration'].unsqueeze(-1))
        
        # One-hot encode intonation [batch_size, seq_len] -> [batch_size, seq_len, 3]
        intonation = F.one_hot(prosody_features['intonation'].long(), num_classes=3).float()
        intonation = self.intonation_embedding(intonation)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=emphasis.device)
        pos_embeddings = self.pos_embedding(positions)
        pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine features [batch_size, seq_len, dim*4]
        combined = torch.cat([
            emphasis,
            pause,
            intonation,
            pos_embeddings
        ], dim=-1)
        
        # Mix features [batch_size, seq_len, dim]
        prosody_encoding = self.prosody_mixer(combined)
        
        # Self-attention
        attn_output, _ = self.prosody_attention(
            query=prosody_encoding,
            key=prosody_encoding,
            value=prosody_encoding,
            key_padding_mask=None,  # Optional mask if needed
            need_weights=False
        )
        
        return attn_output
    
class ProsodyAwareTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()
        
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        
        # Prosody attention
        self.prosody_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward layers
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        
        # Prosody integration
        self.prosody_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                prosody_encoding: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
            t: [batch_size, dim]
            prosody_encoding: [batch_size, seq_len, dim]
            mask: [batch_size, seq_len]
        """
        # Regular attention
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask)
        
        # Prosody attention
        prosody_out, _ = self.prosody_attention(
            query=x,
            key=prosody_encoding,
            value=prosody_encoding,
            key_padding_mask=mask if mask is not None else None,
            need_weights=False
        )
        
        # Gate prosody influence
        prosody_gate = self.prosody_gate(prosody_out)
        
        # Combine attentions
        x = x + gate_msa.unsqueeze(1) * attn_output + prosody_gate * prosody_out
        
        # Feed-forward
        norm = self.ff_norm(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        
        return x
    
class ProsodyAwareDiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        ff_mult=4,
        text_dim=None,
        text_num_embeds=None,
        mel_dim=100,
        conv_layers=4,
    ):
        super().__init__()
        
        self.dim = dim
        
        # Original DiT components
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        if text_num_embeds is None:
            raise ValueError("text_num_embeds must be provided (size of the vocabulary)")
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        
        # Add prosody encoder
        self.prosody_encoder = ProsodyEncoder(dim)
        
        # Modified transformer blocks that accept prosody
        self.transformer_blocks = nn.ModuleList([
            ProsodyAwareTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=0.1
            ) for _ in range(depth)
        ])
        
        # Output layers
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        
        # Prosody conditioning mixer
        self.prosody_mixer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                text: torch.Tensor, time: torch.Tensor,
                prosody_features: Dict[str, torch.Tensor],
                drop_audio_cond=False, drop_text=False,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Shape assertions
        batch_size, seq_len, mel_dim = x.shape
        assert cond.shape[0] == batch_size and cond.shape[1] == seq_len, \
            f"Expected cond shape with batch_size {batch_size} and seq_len {seq_len}, got {cond.shape}"

        assert prosody_features['emphasis'].shape == (batch_size, seq_len), \
            f"Expected emphasis shape {(batch_size, seq_len)}, got {prosody_features['emphasis'].shape}"
        
        # Process inputs
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)
        
        # Get prosody encoding
        prosody_encoding = self.prosody_encoder(prosody_features)
        
        # Mix with input
        x = self.prosody_mixer(torch.cat([x, prosody_encoding], dim=-1))
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, t, prosody_encoding, mask=mask)
        
        return self.proj_out(self.norm_out(x, t))
    
def train_step(model, batch, optimizer):
    # Unpack batch
    x = batch['input']
    cond = batch['condition']
    text = batch['text']
    time = batch['timestep']
    prosody_features = {
        'emphasis': batch['emphasis'],
        'pause_duration': batch['pause_duration'],
        'intonation': batch['intonation']
    }
    
    # Forward pass with prosody
    pred = model(
        x=x,
        cond=cond,
        text=text,
        time=time,
        prosody_features=prosody_features
    )
    
    # Calculate loss with prosody weights
    loss = F.mse_loss(pred, batch['target'])
    
    # Add prosody-specific losses
    prosody_loss = calculate_prosody_loss(
        pred,
        batch['target'],
        prosody_features
    )
    
    total_loss = loss + 0.1 * prosody_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def calculate_prosody_loss(pred: torch.Tensor, 
                         target: torch.Tensor,
                         prosody_features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Calculate prosody-specific losses
    
    Args:
        pred: [batch_size, seq_len, dim]
        target: [batch_size, seq_len, dim]
        prosody_features: Dict with emphasis, pause_duration, intonation
    """
    losses = []
    
    # Emphasis loss - variance should be higher for emphasized regions
    emphasis_mask = prosody_features['emphasis'] > 0
    if emphasis_mask.any():
        emphasized_pred = pred[emphasis_mask]
        emphasized_target = target[emphasis_mask]
        emphasis_loss = F.mse_loss(
            pred[emphasis_mask],
            target[emphasis_mask],
            reduction='mean'
        )
        losses.append(emphasis_loss)
    
    # Pause loss - ensure proper duration
    pause_mask = prosody_features['pause_duration'] > 0
    if pause_mask.any():
        pause_loss = F.mse_loss(
            pred[pause_mask],
            torch.zeros_like(pred[pause_mask]),
            reduction='mean'
        )
        losses.append(pause_loss)
    
    # Intonation loss
    question_mask = prosody_features['intonation'] == 1
    if question_mask.any():
        question_pred = pred[question_mask]  # Shape: [N, dim]
        seq_len = question_pred.size(1)
        rising_target = torch.linspace(0, 1, steps=seq_len, device=pred.device).unsqueeze(0)
        rising_target = rising_target.expand_as(question_pred)
        question_loss = F.mse_loss(question_pred, rising_target)
        losses.append(question_loss)

    
    return sum(losses) if losses else torch.tensor(0.0, device=pred.device)