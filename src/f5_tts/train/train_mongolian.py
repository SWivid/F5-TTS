import os
import sys
from importlib.resources import files
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from cached_path import cached_path
from torch.utils.tensorboard import SummaryWriter


import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from f5_tts.mongolian.text_processor import MongolianTextProcessor
from f5_tts.mongolian.dataset import MongolianAwareDataset, collate_fn
from f5_tts.mongolian.model import ProsodyAwareDiT
from f5_tts.model.utils import get_tokenizer
from f5_tts.model import CFM


# Training settings
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"

def setup_wandb(args, accelerator):
    if accelerator.is_main_process:
        wandb_run_name = f"mongolian_{args.exp_name}"
        wandb_project = "mongolian_tts"
        
        if args.wandb_resume_id:
            init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": args.wandb_resume_id}}
        else:
            init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

        accelerator.init_trackers(
            project_name=wandb_project,
            init_kwargs=init_kwargs,
            config={
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "num_warmup_updates": args.num_warmup_updates,
                "batch_size": args.batch_size,
                "grad_accumulation_steps": args.grad_accumulation_steps,
                "max_grad_norm": args.max_grad_norm,
                "gpus": accelerator.num_processes,
            },
        )


def load_dataset(dataset_path):
    data = []
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            filename, text = line.strip().split('|')
            audio_path = os.path.join(dataset_path, 'wavs', filename + '.wav')
            data.append({'filename': filename, 'text': text, 'audio_path': audio_path})
    return data


def setup_model(vocab_size, model_dim=1024, model_depth=22, n_mel_channels=100):
    # Initialize model with Mongolian-specific configuration
    model_cfg = dict(
        dim=model_dim,
        depth=model_depth,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
        mel_dim=n_mel_channels,
        text_num_embeds=vocab_size  # Pass vocab_size here
    )
    
    transformer = ProsodyAwareDiT(**model_cfg)
    
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=dict(
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=n_mel_channels,
            target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        odeint_kwargs=dict(
            method="euler",
        )
    )
    
    return model


def train_epoch(model, train_loader, optimizer, scheduler, accelerator, args, tb_writer, epoch):
    model.train()
    total_loss = 0
    global_step = epoch * len(train_loader)  # Initialize global step for TensorBoard
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch+1}",
        disable=not accelerator.is_local_main_process
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            # Move tensors to the correct device
            mel = batch['mel'].to(accelerator.device)                  # Shape: [batch_size, n_mels, max_mel_length]
            text = batch['text'].to(accelerator.device)                # Shape: [batch_size, max_token_len]
            mel_lengths = batch['mel_lengths'].to(accelerator.device)  # Shape: [batch_size]
            prosody_features = {
                'emphasis': batch['prosody_features']['emphasis'].to(accelerator.device),
                'pause_duration': batch['prosody_features']['pause_duration'].to(accelerator.device),
                'intonation': batch['prosody_features']['intonation'].to(accelerator.device)
            }

            # Forward pass
            loss, cond, pred = model(
                inp=mel.permute(0, 2, 1),  # Adjust shape to [batch_size, max_mel_length, n_mels] if needed
                text=text,
                lens=mel_lengths,
                prosody_features=prosody_features
            )
            
            # Backward pass
            accelerator.backward(loss)
            
            if args.max_grad_norm > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Increment global step
            global_step += 1

            # Log metrics to TensorBoard
            if accelerator.is_local_main_process:
                tb_writer.add_scalar("Loss/train", loss.item(), global_step)
                tb_writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], global_step)
            
            total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(train_loader)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def main(args):
    # Setup accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    
    # Setup logging
    setup_wandb(args, accelerator)
    
    # Initialize TensorBoard writer
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    train_dataset = MongolianAwareDataset(dataset=dataset)

    # Get vocabulary size from the dataset
    vocab_size = len(train_dataset.vocab)
    
    # Setup data loader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    
    # Setup model
    model = setup_model(vocab_size)
    
    # Load checkpoint if finetuning
    if args.finetune and args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = args.num_warmup_updates * accelerator.num_processes
    
    if warmup_steps >= total_steps:
        warmup_steps = int(0.1 * total_steps)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='linear'
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        
        tb_writer = SummaryWriter(log_dir=args.output_dir)
        
        avg_loss = train_epoch(
             model=model,
            train_loader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            args=args,
            tb_writer=tb_writer,
            epoch=epoch
        )
        
        # Save checkpoint
        if accelerator.is_main_process:
            # Ensure output directory exists
            os.makedirs(args.output_dir, exist_ok=True)
            
            

            checkpoint = {
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    checkpoint,
                    os.path.join(args.output_dir, f"model_best.pt")
                )

            # Regular checkpoint
            if epoch % args.save_frequency == 0:
                torch.save(
                    checkpoint,
                    os.path.join(args.output_dir, f"model_{epoch}.pt")
                )

        accelerator.wait_for_everyone()
    # Save vocabulary after training
    if accelerator.is_main_process:
        vocab_path = os.path.join(args.output_dir, "vocab.txt")
        print(f"Saving vocabulary to {vocab_path}")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for token in train_dataset.vocab.idx2token.values():
                f.write(f"{token}\n")
                
    accelerator.end_training()
    tb_writer.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    #parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_warmup_updates", type=int, default=1000)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_frequency", type=int, default=5)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--checkpoint_path", type=str)
    #parser.add_argument("--batch_size_type", choices=["frame", "sample"], default="frame")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--wandb_resume_id", type=str)
    parser.add_argument("--exp_name", type=str, default="mongolian_tts")
    parser.add_argument("--model_dim", type=int, default=1024)
    parser.add_argument("--model_depth", type=int, default=22)
    
    args = parser.parse_args()
    
    main(args)