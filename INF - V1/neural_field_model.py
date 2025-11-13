"""
Neural Field 3D Image Restoration - Model and Utilities
Combines all model definitions and utility functions in one file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import tifffile


# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_tif_volume(filepath):
    """Load 3D volume from TIF file"""
    volume = tifffile.imread(filepath)
    
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    elif volume.ndim == 4:
        if volume.shape[-1] <= 4:
            volume = volume[..., 0]
    
    return volume.astype(np.float32)


def normalize_volume(volume, percentile_clip=True, lower=1, upper=99):
    """Normalize volume to [0, 1] range"""
    volume = volume.astype(np.float32)
    
    if percentile_clip:
        vmin = np.percentile(volume, lower)
        vmax = np.percentile(volume, upper)
    else:
        vmin = volume.min()
        vmax = volume.max()
    
    if vmax - vmin < 1e-6:
        return np.zeros_like(volume)
    
    volume = np.clip(volume, vmin, vmax)
    volume = (volume - vmin) / (vmax - vmin)
    
    return volume


def save_tif_volume(volume, filepath):
    """Save 3D volume to TIF file"""
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    if volume.max() <= 1.0:
        volume = (volume * 65535).astype(np.uint16)
    else:
        volume = volume.astype(np.uint16)
    
    tifffile.imwrite(filepath, volume, compression='zlib')


def create_foreground_mask(volume, threshold=0.1):
    """Create binary foreground mask"""
    if isinstance(volume, torch.Tensor):
        return (volume > threshold).float()
    else:
        return (volume > threshold).astype(np.float32)


# ============================================================================
# SAMPLING UTILITIES
# ============================================================================

def sample_block(volume_shape, block_size, foreground_mask=None, 
                 min_foreground_ratio=0.01, max_attempts=10):
    """Sample a random block location from volume"""
    D, H, W = volume_shape
    B = block_size
    
    z_max = max(1, D - B + 1)
    y_max = max(1, H - B + 1)
    x_max = max(1, W - B + 1)
    
    if foreground_mask is None:
        z0 = np.random.randint(0, z_max)
        y0 = np.random.randint(0, y_max)
        x0 = np.random.randint(0, x_max)
        return z0, y0, x0
    
    for _ in range(max_attempts):
        z0 = np.random.randint(0, z_max)
        y0 = np.random.randint(0, y_max)
        x0 = np.random.randint(0, x_max)
        
        block_mask = foreground_mask[z0:z0+B, y0:y0+B, x0:x0+B]
        fg_ratio = block_mask.mean()
        
        if fg_ratio > min_foreground_ratio:
            return z0, y0, x0
    
    return z0, y0, x0


def sample_voxels_from_block(block_shape, foreground_mask, 
                              num_samples, foreground_ratio=0.75):
    """Sample voxel coordinates from a block"""
    D, H, W = block_shape
    
    zz, yy, xx = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing='ij'
    )
    idx_all = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)
    
    mask_flat = foreground_mask.reshape(-1)
    fg_indices = idx_all[mask_flat > 0.1]
    bg_indices = idx_all[mask_flat <= 0.1]
    
    num_fg = int(num_samples * foreground_ratio)
    num_bg = num_samples - num_fg
    
    if len(fg_indices) > 0 and num_fg > 0:
        if len(fg_indices) >= num_fg:
            fg_sample_idx = np.random.choice(len(fg_indices), num_fg, replace=False)
            fg_sample = fg_indices[fg_sample_idx]
        else:
            fg_sample_idx = np.random.choice(len(fg_indices), num_fg, replace=True)
            fg_sample = fg_indices[fg_sample_idx]
    else:
        fg_sample = idx_all[np.random.choice(len(idx_all), num_fg, replace=False)]
    
    if len(bg_indices) > 0 and num_bg > 0:
        if len(bg_indices) >= num_bg:
            bg_sample_idx = np.random.choice(len(bg_indices), num_bg, replace=False)
            bg_sample = bg_indices[bg_sample_idx]
        else:
            bg_sample_idx = np.random.choice(len(bg_indices), num_bg, replace=True)
            bg_sample = bg_indices[bg_sample_idx]
    else:
        bg_sample = idx_all[np.random.choice(len(idx_all), num_bg, replace=False)]
    
    samples = np.concatenate([fg_sample, bg_sample], axis=0)
    perm = np.random.permutation(len(samples))
    samples = samples[perm]
    
    return samples


def coords_to_normalized(coords_ijk, block_offset, volume_shape, block_size=None):
    """Convert voxel indices to normalized coordinates in [-1, 1]
    
    Returns both global and local (block-relative) coordinates
    """
    z0, y0, x0 = block_offset
    D, H, W = volume_shape
    
    # Global coordinates (relative to full volume)
    coords_ijk_global = coords_ijk + np.array([z0, y0, x0])
    
    z_norm_global = coords_ijk_global[:, 0] / (D - 1) * 2 - 1
    y_norm_global = coords_ijk_global[:, 1] / (H - 1) * 2 - 1
    x_norm_global = coords_ijk_global[:, 2] / (W - 1) * 2 - 1
    
    coords_norm_global = np.stack([x_norm_global, y_norm_global, z_norm_global], axis=-1)
    
    # Local coordinates (relative to block) - critical for local encoder!
    if block_size is not None:
        # coords_ijk is already in block-local space [0, block_size)
        # Normalize to [-1, 1]
        z_norm_local = coords_ijk[:, 0] / (block_size - 1) * 2 - 1 if block_size > 1 else 0
        y_norm_local = coords_ijk[:, 1] / (block_size - 1) * 2 - 1 if block_size > 1 else 0
        x_norm_local = coords_ijk[:, 2] / (block_size - 1) * 2 - 1 if block_size > 1 else 0
        
        coords_norm_local = np.stack([x_norm_local, y_norm_local, z_norm_local], axis=-1)
        
        # Concatenate global and local coordinates
        coords_norm = np.concatenate([coords_norm_global, coords_norm_local], axis=-1)
    else:
        coords_norm = coords_norm_global
    
    return coords_norm.astype(np.float32)


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class GlobalEncoder3D(nn.Module):
    """Extract global context from downsampled volume"""
    def __init__(self, in_channels=1, base_channels=16, latent_dim=64):
        super().__init__()
        ch = base_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, ch, 3, padding=1),
            nn.GroupNorm(4, ch),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch, ch*2, 3, stride=2, padding=1),
            nn.GroupNorm(4, ch*2),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(ch*2, ch*4, 3, stride=2, padding=1),
            nn.GroupNorm(4, ch*4),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(ch*4, ch*4, 3, padding=1),
            nn.GroupNorm(4, ch*4),
            nn.SiLU(),
        )
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(ch*4, latent_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x).view(x.size(0), -1)
        z = self.fc(x)
        return z


class LocalEncoder3D(nn.Module):
    """Extract local features from block"""
    def __init__(self, in_channels=1, base_channels=16, latent_dim=64):
        super().__init__()
        ch = base_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, ch, 3, padding=1),
            nn.GroupNorm(4, ch),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(ch, ch*2, 3, stride=2, padding=1),
            nn.GroupNorm(4, ch*2),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(ch*2, ch*4, 3, stride=2, padding=1),
            nn.GroupNorm(4, ch*4),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(ch*4, ch*4, 3, padding=1),
            nn.GroupNorm(4, ch*4),
            nn.SiLU(),
        )
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(ch*4, latent_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x).view(x.size(0), -1)
        z = self.fc(x)
        return z


class CoordEncoder(nn.Module):
    """Fourier feature encoding for 3D coordinates
    
    Supports both 3D (global only) and 6D (global + local) coordinates
    """
    def __init__(self, num_freqs=6):
        super().__init__()
        self.num_freqs = num_freqs
        
    def forward(self, coords):
        N, coord_dim = coords.shape
        device = coords.device
        
        freqs = torch.arange(self.num_freqs, device=device).float()
        freqs = (2.0 ** freqs) * math.pi
        
        def encode_dim(dim):
            x = coords[:, dim:dim+1]
            x = x * freqs[None, :]
            return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        
        # Encode all coordinate dimensions (3 or 6)
        encodings = []
        for dim in range(coord_dim):
            encodings.append(encode_dim(dim))
        
        out = torch.cat([coords] + encodings, dim=-1)
        return out
    
    def get_output_dim(self, input_dim=3):
        """Get output dimension based on input coordinate dimension"""
        return input_dim + input_dim * 2 * self.num_freqs
    
    @property
    def output_dim(self):
        # Default to 3D global coordinates for backward compatibility
        return self.get_output_dim(3)


class NeuralFieldMLP(nn.Module):
    """Neural Field MLP for voxel-wise prediction"""
    def __init__(self, coord_dim, local_dim, global_dim, 
                 hidden_dim=128, num_layers=4, residual_mode=True,
                 residual_scale=0.2):
        super().__init__()
        self.residual_mode = residual_mode
        self.residual_scale = residual_scale  # Scale factor for bounded residual
        
        in_dim = coord_dim + local_dim + global_dim
        
        layers = []
        d = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, coord_feat, local_latent, global_latent, degraded_values=None):
        N = coord_feat.size(0)
        
        # Expand latents to match number of query points
        # Don't use [0] - handle batch dimension properly
        if local_latent.dim() == 2:  # [batch, dim]
            L = local_latent.expand(N, -1)
        else:  # [batch=1, dim]
            L = local_latent.squeeze(0).unsqueeze(0).expand(N, -1)
        
        if global_latent.dim() == 2:  # [batch, dim]
            G = global_latent.expand(N, -1)
        else:  # [batch=1, dim]
            G = global_latent.squeeze(0).unsqueeze(0).expand(N, -1)
        
        x = torch.cat([coord_feat, L, G], dim=-1)
        
        h = self.mlp(x)
        r = self.out(h).squeeze(-1)
        
        if self.residual_mode:
            assert degraded_values is not None
            if degraded_values.dim() > 1:
                degraded_values = degraded_values.squeeze(-1)
            # Apply tanh and scale to bound residual range
            r_bounded = self.residual_scale * torch.tanh(r)
            return degraded_values + r_bounded
        else:
            return r


class NeuralFieldRefiner(nn.Module):
    """Complete Neural Field 3D Image Restoration model"""
    def __init__(self,
                 num_freqs=6,
                 local_latent_dim=64,
                 global_latent_dim=64,
                 hidden_dim=128,
                 mlp_layers=4,
                 local_base_channels=16,
                 global_base_channels=16,
                 use_local_coords=True):
        super().__init__()
        
        self.use_local_coords = use_local_coords
        
        self.global_encoder = GlobalEncoder3D(
            in_channels=1, 
            base_channels=global_base_channels, 
            latent_dim=global_latent_dim
        )
        
        self.local_encoder = LocalEncoder3D(
            in_channels=1, 
            base_channels=local_base_channels, 
            latent_dim=local_latent_dim
        )
        
        self.coord_encoder = CoordEncoder(num_freqs=num_freqs)
        # Coordinate dimension: 6D (global + local) if use_local_coords else 3D
        coord_input_dim = 6 if use_local_coords else 3
        coord_dim = self.coord_encoder.get_output_dim(coord_input_dim)
        
        self.field_mlp = NeuralFieldMLP(
            coord_dim=coord_dim,
            local_dim=local_latent_dim,
            global_dim=global_latent_dim,
            hidden_dim=hidden_dim,
            num_layers=mlp_layers,
            residual_mode=True,
            residual_scale=0.2,  # Bounded residual range
        )
        
    def encode_global(self, volume_down):
        return self.global_encoder(volume_down)
    
    def encode_local(self, block):
        return self.local_encoder(block)
    
    def query_field(self, coords, local_latent, global_latent, degraded_values):
        coord_feat = self.coord_encoder(coords)
        pred = self.field_mlp(coord_feat, local_latent, global_latent, degraded_values)
        return pred
    
    def forward(self, volume_deg, block_deg, coords, degraded_values, 
                global_target_shape=(32, 32, 32)):
        """Forward pass (not used in training/inference, but kept for compatibility)
        
        Note: Use explicit encode_global/encode_local/query_field calls instead
        to ensure consistency with training/inference pipeline.
        """
        volume_down = F.interpolate(
            volume_deg, 
            size=global_target_shape,  # Use fixed target shape, not scale_factor
            mode="trilinear", 
            align_corners=False
        )
        
        global_latent = self.encode_global(volume_down)
        local_latent = self.encode_local(block_deg)
        pred = self.query_field(coords, local_latent, global_latent, degraded_values)
        
        return pred

