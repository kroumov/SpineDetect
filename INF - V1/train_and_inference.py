"""
Neural Field 3D Image Restoration - Training and Inference
Combined training and inference functionality
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from scipy.signal import windows

from config import Config
from neural_field_model import (
    NeuralFieldRefiner, load_tif_volume, normalize_volume, save_tif_volume,
    create_foreground_mask, sample_block, sample_voxels_from_block, coords_to_normalized
)


# ============================================================================
# DATASET
# ============================================================================

class VolumeDataset:
    """Dataset for 3D volume restoration"""
    def __init__(self, data_dir, use_simulation=True, noise_std=0.03):
        self.data_dir = Path(data_dir)
        self.use_simulation = use_simulation
        self.noise_std = noise_std  # Configurable noise level
        
        self.volume_files = sorted(list(self.data_dir.glob("*.tif")) + 
                                   list(self.data_dir.glob("*.tiff")))
        
        if len(self.volume_files) == 0:
            raise ValueError(f"No TIF files found in {data_dir}")
        
        print(f"Found {len(self.volume_files)} volume files")
        
    def __len__(self):
        return len(self.volume_files)
    
    def load_volume(self, idx):
        filepath = self.volume_files[idx]
        volume = load_tif_volume(str(filepath))
        volume_clean = normalize_volume(volume)
        
        if self.use_simulation:
            volume_deg = self.simulate_degradation(volume_clean)
        else:
            volume_deg = volume_clean.copy()
        
        return volume_deg, volume_clean
    
    def simulate_degradation(self, volume_clean):
        """Simulate image degradation with noise
        
        Uses gentler noise (0.03 instead of 0.05) to encourage conservative learning
        """
        volume_deg = volume_clean.copy()
        noise = np.random.randn(*volume_deg.shape) * self.noise_std
        volume_deg = volume_deg + noise
        volume_deg = np.clip(volume_deg, 0, 1)
        return volume_deg


# ============================================================================
# TRAINING
# ============================================================================

def train_step(model, volume_deg, volume_clean, optimizer, config, global_latent=None):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    device = next(model.parameters()).device
    D, H, W = volume_deg.shape[-3:]
    B = config.BLOCK_SIZE
    K = config.NUM_VOXEL_SAMPLES
    
    # Encode global context
    if global_latent is None:
        # Use fixed target shape instead of scale_factor for stability
        volume_down = F.interpolate(
            volume_deg.to(device), 
            size=config.GLOBAL_TARGET_SHAPE,
            mode="trilinear",
            align_corners=False
        )
        global_latent = model.encode_global(volume_down)
    
    # Create foreground mask from DEGRADED volume (not clean!)
    # This ensures training-inference consistency
    with torch.no_grad():
        fg_mask = (volume_deg[0, 0] > config.FOREGROUND_THRESHOLD).cpu().numpy()
    
    # Sample a block
    z0, y0, x0 = sample_block(
        volume_shape=(D, H, W),
        block_size=B,
        foreground_mask=fg_mask,
        min_foreground_ratio=0.01
    )
    
    # Extract blocks
    block_deg = volume_deg[:, :, z0:z0+B, y0:y0+B, x0:x0+B]
    block_clean = volume_clean[:, :, z0:z0+B, y0:y0+B, x0:x0+B]
    block_mask = fg_mask[z0:z0+B, y0:y0+B, x0:x0+B]
    
    # Get actual block shape
    actual_block_shape = block_deg.shape[2:]
    
    # Pad block if necessary (BEFORE encoding and sampling)
    if actual_block_shape != (B, B, B):
        pad_d = B - actual_block_shape[0]
        pad_h = B - actual_block_shape[1]
        pad_w = B - actual_block_shape[2]
        
        # Pad all blocks to ensure consistency
        block_deg = torch.nn.functional.pad(
            block_deg, (0, pad_w, 0, pad_h, 0, pad_d), mode='replicate'
        )
        block_clean = torch.nn.functional.pad(
            block_clean, (0, pad_w, 0, pad_h, 0, pad_d), mode='replicate'
        )
        block_mask = np.pad(
            block_mask, 
            ((0, pad_d), (0, pad_h), (0, pad_w)), 
            mode='edge'
        )
    
    # Move to device AFTER padding
    block_deg = block_deg.to(device)
    block_clean = block_clean.to(device)
    
    # Encode local features (now using padded block)
    local_latent = model.encode_local(block_deg)
    
    # Sample voxels from PADDED block
    samples = sample_voxels_from_block(
        block_shape=(B, B, B),  # Use padded size
        foreground_mask=block_mask,  # Use padded mask
        num_samples=K,
        foreground_ratio=config.FOREGROUND_RATIO
    )
    
    # Convert to normalized coordinates (both global and local)
    coords_norm = coords_to_normalized(
        coords_ijk=samples,
        block_offset=(z0, y0, x0),
        volume_shape=(D, H, W),
        block_size=B  # Include local coordinates
    )
    coords_norm = torch.from_numpy(coords_norm).to(device)
    
    # Get voxel values from PADDED blocks
    zz, yy, xx = samples[:, 0], samples[:, 1], samples[:, 2]
    deg_vals = block_deg[0, 0, zz, yy, xx]
    clean_vals = block_clean[0, 0, zz, yy, xx]
    
    # Predict
    pred_vals = model.query_field(coords_norm, local_latent, global_latent, deg_vals)
    
    # Compute weighted loss (use padded mask)
    mask_fg_vals = block_mask[zz, yy, xx] > 0.1
    mask_fg = torch.from_numpy(mask_fg_vals).to(device)
    weights = torch.where(
        mask_fg,
        torch.full_like(clean_vals, config.WEIGHT_FOREGROUND),
        torch.full_like(clean_vals, config.WEIGHT_BACKGROUND)
    )
    
    # Main reconstruction loss
    mse = (pred_vals - clean_vals) ** 2
    loss = (weights * mse).mean()
    
    # Conservative regularization: penalize large deviations from input
    # This encourages the model to be conservative and not change too much
    if config.USE_CONSERVATIVE_REG:
        reg_loss = ((pred_vals - deg_vals) ** 2).mean()
        loss = loss + config.CONSERVATIVE_REG_WEIGHT * reg_loss
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train(config):
    """Main training loop"""
    config.create_dirs()
    
    print("Initializing model...")
    model = NeuralFieldRefiner(
        num_freqs=config.NUM_FREQS,
        local_latent_dim=config.LOCAL_LATENT_DIM,
        global_latent_dim=config.GLOBAL_LATENT_DIM,
        hidden_dim=config.MLP_HIDDEN_DIM,
        mlp_layers=config.MLP_NUM_LAYERS,
        local_base_channels=config.LOCAL_BASE_CHANNELS,
        global_base_channels=config.GLOBAL_BASE_CHANNELS,
        use_local_coords=True,
    )
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model on device: {device}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    print("Loading dataset...")
    dataset = VolumeDataset(
        config.TRAIN_DATA_DIR, 
        use_simulation=True,
        noise_std=config.DEGRADATION_NOISE_STD
    )
    
    writer = SummaryWriter(config.LOG_DIR)
    
    print("Starting training...")
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        vol_idx = np.random.randint(0, len(dataset))
        volume_deg_np, volume_clean_np = dataset.load_volume(vol_idx)
        
        volume_deg = torch.from_numpy(volume_deg_np[None, None, ...]).float()
        volume_clean = torch.from_numpy(volume_clean_np[None, None, ...]).float()
        
        print(f"Training on volume {vol_idx}: {dataset.volume_files[vol_idx].name}")
        print(f"Volume shape: {volume_deg.shape}")
        
        # Note: global_latent will be computed per-step now (fix #8)
        # This pre-computation is removed to ensure consistency
        global_latent = None  # Will be computed in each train_step
        
        epoch_losses = []
        pbar = tqdm(range(config.STEPS_PER_EPOCH), desc="Training")
        
        for step in pbar:
            loss = train_step(
                model, volume_deg, volume_clean, 
                optimizer, config, global_latent
            )
            epoch_losses.append(loss)
            
            if step % config.LOG_INTERVAL == 0:
                pbar.set_postfix({'loss': f'{loss:.6f}'})
                writer.add_scalar('train/loss_step', loss, global_step)
            
            global_step += 1
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
        writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"model_epoch_{epoch+1:03d}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    final_path = os.path.join(config.CHECKPOINT_DIR, "model_final.pth")
    torch.save({
        'epoch': config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"Saved final model: {final_path}")
    
    writer.close()
    print("Training complete!")


# ============================================================================
# INFERENCE
# ============================================================================

def create_tukey_window_3d(size, alpha=0.5, min_value=1e-3):
    """Create 3D Tukey window for smooth blending"""
    tukey_1d = windows.tukey(size, alpha=alpha)
    window = tukey_1d[:, None, None] * tukey_1d[None, :, None] * tukey_1d[None, None, :]
    window = window.astype(np.float32)
    window = np.clip(window, min_value, None)
    window /= window.max()
    return window


def inference_on_block(model, block_deg, local_latent, global_latent,
                       block_offset, volume_shape, chunk_size, device,
                       actual_block_size=None, block_size_fixed=64):
    """Inference on a single block"""
    B = block_deg.shape[0]
    
    if actual_block_size is not None:
        D_block, H_block, W_block = actual_block_size
    else:
        D_block, H_block, W_block = B, B, B
    
    # Generate voxel coordinates (only for actual region)
    zz, yy, xx = np.meshgrid(
        np.arange(D_block), np.arange(H_block), np.arange(W_block), indexing='ij'
    )
    coords_ijk = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
    
    # For inference, use FIXED block_size for local coords (same as training)
    # This ensures training-inference consistency even for edge blocks
    coords_norm = coords_to_normalized(coords_ijk, block_offset, volume_shape, 
                                       block_size=block_size_fixed)
    coords_norm = torch.from_numpy(coords_norm).to(device)
    
    deg_vals = block_deg[:D_block, :H_block, :W_block].ravel()
    deg_vals = torch.from_numpy(deg_vals).to(device)
    
    # Process in chunks
    num_voxels = len(coords_norm)
    pred_vals = []
    
    for i in range(0, num_voxels, chunk_size):
        chunk_coords = coords_norm[i:i+chunk_size]
        chunk_deg = deg_vals[i:i+chunk_size]
        
        chunk_pred = model.query_field(
            chunk_coords, local_latent, global_latent, chunk_deg
        )
        pred_vals.append(chunk_pred.cpu().numpy())
    
    pred_vals = np.concatenate(pred_vals, axis=0)
    block_pred = pred_vals.reshape(D_block, H_block, W_block)
    
    # Pad back if necessary
    if D_block < B or H_block < B or W_block < B:
        block_pred_padded = np.zeros((B, B, B), dtype=np.float32)
        block_pred_padded[:D_block, :H_block, :W_block] = block_pred
        block_pred = block_pred_padded
    
    return block_pred


def inference_on_volume(model, volume_deg, config, device):
    """Perform inference on full volume using sliding window"""
    model.eval()
    
    D, H, W = volume_deg.shape
    B = config.INFERENCE_BLOCK_SIZE
    overlap = config.INFERENCE_OVERLAP
    stride = B - overlap
    chunk_size = config.INFERENCE_CHUNK_SIZE
    
    volume_deg_tensor = torch.from_numpy(volume_deg[None, None, ...])
    
    output_volume = np.zeros_like(volume_deg)
    weight_volume = np.zeros_like(volume_deg)
    
    if getattr(config, "USE_HANN_WINDOW", False):
        window = create_tukey_window_3d(
            B, alpha=getattr(config, "WINDOW_ALPHA", 0.5),
            min_value=getattr(config, "WINDOW_MIN_VALUE", 1e-3)
        )
    else:
        window = np.ones((B, B, B), dtype=np.float32)
    
    print("Computing global context...")
    with torch.no_grad():
        # Use fixed target shape for stability
        volume_down = F.interpolate(
            volume_deg_tensor.to(device),
            size=config.GLOBAL_TARGET_SHAPE,
            mode="trilinear",
            align_corners=False
        )
        global_latent = model.encode_global(volume_down)
    
    # Generate block positions
    z_starts = list(range(0, D - B + 1, stride)) + [max(0, D - B)]
    y_starts = list(range(0, H - B + 1, stride)) + [max(0, H - B)]
    x_starts = list(range(0, W - B + 1, stride)) + [max(0, W - B)]
    
    z_starts = sorted(list(set(z_starts)))
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))
    
    total_blocks = len(z_starts) * len(y_starts) * len(x_starts)
    print(f"Processing {total_blocks} blocks...")
    
    with torch.no_grad():
        pbar = tqdm(total=total_blocks, desc="Inference")
        
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    z_end = min(z0 + B, D)
                    y_end = min(y0 + B, H)
                    x_end = min(x0 + B, W)
                    
                    block_deg = volume_deg[z0:z_end, y0:y_end, x0:x_end]
                    
                    # Pad if necessary
                    pad_z = B - (z_end - z0)
                    pad_y = B - (y_end - y0)
                    pad_x = B - (x_end - x0)
                    
                    if pad_z > 0 or pad_y > 0 or pad_x > 0:
                        block_deg = np.pad(
                            block_deg,
                            ((0, pad_z), (0, pad_y), (0, pad_x)),
                            mode='reflect'
                        )
                    
                    block_deg_tensor = torch.from_numpy(block_deg[None, None, ...]).to(device)
                    local_latent = model.encode_local(block_deg_tensor)
                    
                    actual_block_size = (z_end - z0, y_end - y0, x_end - x0)
                    block_pred = inference_on_block(
                        model, block_deg, local_latent, global_latent,
                        (z0, y0, x0), (D, H, W), chunk_size, device,
                        actual_block_size=actual_block_size,
                        block_size_fixed=B  # Use fixed B for consistent local coords
                    )
                    
                    # Remove padding
                    if pad_z > 0 or pad_y > 0 or pad_x > 0:
                        block_pred = block_pred[:z_end-z0, :y_end-y0, :x_end-x0]
                        block_window = window[:z_end-z0, :y_end-y0, :x_end-x0]
                    else:
                        block_window = window
                    
                    # Accumulate
                    output_volume[z0:z_end, y0:y_end, x0:x_end] += block_pred * block_window
                    weight_volume[z0:z_end, y0:y_end, x0:x_end] += block_window
                    
                    pbar.update(1)
        
        pbar.close()
    
    weight_volume = np.maximum(weight_volume, 1e-8)
    volume_clean = output_volume / weight_volume
    
    # Conservative blending: optionally mix with input for extra safety
    if config.USE_CONSERVATIVE_BLEND:
        alpha = config.CONSERVATIVE_BLEND_ALPHA
        volume_clean = alpha * volume_clean + (1 - alpha) * volume_deg
        print(f"Applied conservative blending with alpha={alpha:.2f}")
    
    volume_clean = np.clip(volume_clean, 0, 1)
    
    return volume_clean


def run_inference(input_path, output_path, checkpoint_path, device='cuda'):
    """Run inference on a single volume"""
    config = Config()
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = NeuralFieldRefiner(
        num_freqs=config.NUM_FREQS,
        local_latent_dim=config.LOCAL_LATENT_DIM,
        global_latent_dim=config.GLOBAL_LATENT_DIM,
        hidden_dim=config.MLP_HIDDEN_DIM,
        mlp_layers=config.MLP_NUM_LAYERS,
        local_base_channels=config.LOCAL_BASE_CHANNELS,
        global_base_channels=config.GLOBAL_BASE_CHANNELS,
        use_local_coords=True,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    print(f"Loading input volume: {input_path}")
    volume_deg = load_tif_volume(input_path)
    print(f"Input shape: {volume_deg.shape}")
    
    volume_deg = normalize_volume(volume_deg)
    
    print("Running inference...")
    volume_clean = inference_on_volume(model, volume_deg, config, device)
    
    print(f"Saving output to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_tif_volume(volume_clean, output_path)
    
    print("Inference complete!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Neural Field 3D Image Restoration')
    parser.add_argument('mode', choices=['train', 'inference'], 
                        help='Mode: train or inference')
    parser.add_argument('--input', type=str, help='Input TIF file (inference mode)')
    parser.add_argument('--output', type=str, help='Output TIF file (inference mode)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path (inference mode)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        config = Config()
        train(config)
    elif args.mode == 'inference':
        if not all([args.input, args.output, args.checkpoint]):
            parser.error("inference mode requires --input, --output, and --checkpoint")
        run_inference(args.input, args.output, args.checkpoint, args.device)


if __name__ == "__main__":
    main()

