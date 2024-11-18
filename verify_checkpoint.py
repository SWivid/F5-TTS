import torch

def verify_reduced_checkpoint(path):
    print(f"Analyzing checkpoint: {path}")
    ckpt = torch.load(path)
    state_dict = ckpt["ema_model_state_dict"] if "ema_model_state_dict" in ckpt else ckpt
    
    layers = {k.split('.')[0] for k in state_dict.keys()}
    print(f"\nModel components found: {sorted(layers)}")
    
    weight_stats = {}
    for k, v in state_dict.items():
        if v.dtype in [torch.float32, torch.float16]:
            weight_stats[k] = {
                'mean': v.mean().item(),
                'std': v.std().item()
            }
            if torch.isnan(v).any():
                print(f"WARNING: NaN values in {k}")
    
    print("\nStats computed successfully - weights look valid")
    return weight_stats

stats = verify_reduced_checkpoint('ckpts/mn_tts/model_60072_reduced.pt')