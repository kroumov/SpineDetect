"""
Input/Output Utilities for 3D TIF Image Processing
Handles loading, saving, and visualization of 3D microscopy data
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, Union


def load_tif_volume(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load 3D volume from TIF file
    
    Args:
        filepath: Path to TIF file
        
    Returns:
        volume: 3D numpy array (Z, H, W) in float32
    """
    volume = tifffile.imread(str(filepath))
    
    # Handle different input formats
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    elif volume.ndim == 4:
        if volume.shape[-1] <= 4:
            # RGBA or multi-channel, take first channel
            volume = volume[..., 0]
    
    return volume.astype(np.float32)


def save_tif_volume(volume: np.ndarray, filepath: Union[str, Path], 
                    bit_depth: int = 16) -> None:
    """
    Save 3D volume to TIF file
    
    Args:
        volume: 3D numpy array
        filepath: Output path
        bit_depth: 8 or 16 bit
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if volume.max() <= 1.0:
        # Normalize to bit depth
        if bit_depth == 16:
            volume_save = (volume * 65535).astype(np.uint16)
        else:
            volume_save = (volume * 255).astype(np.uint8)
    else:
        if bit_depth == 16:
            volume_save = volume.astype(np.uint16)
        else:
            volume_save = volume.astype(np.uint8)
    
    tifffile.imwrite(str(filepath), volume_save, compression='zlib')
    print(f"Saved volume to: {filepath}")


def normalize_volume(volume: np.ndarray, percentile_clip: bool = True,
                     lower: float = 1, upper: float = 99) -> np.ndarray:
    """
    Normalize volume to [0, 1] range
    
    Args:
        volume: Input volume
        percentile_clip: Use percentile clipping
        lower: Lower percentile
        upper: Upper percentile
        
    Returns:
        Normalized volume
    """
    volume = volume.astype(np.float32)
    
    if percentile_clip:
        vmin = np.percentile(volume, lower)
        vmax = np.percentile(volume, upper)
    else:
        vmin = volume.min()
        vmax = volume.max()
    
    if vmax - vmin < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    
    volume = np.clip(volume, vmin, vmax)
    volume = (volume - vmin) / (vmax - vmin)
    
    return volume


def visualize_slices(volume: np.ndarray, slice_indices: Optional[list] = None,
                     num_slices: int = 5, title: str = "Volume Slices",
                     save_path: Optional[str] = None) -> None:
    """
    Visualize selected slices from 3D volume
    
    Args:
        volume: 3D volume (Z, H, W)
        slice_indices: Specific indices to visualize
        num_slices: Number of evenly spaced slices if indices not provided
        title: Plot title
        save_path: Path to save figure
    """
    D = volume.shape[0]
    
    if slice_indices is None:
        slice_indices = np.linspace(0, D-1, num_slices, dtype=int)
    
    num_slices = len(slice_indices)
    cols = min(5, num_slices)
    rows = (num_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if num_slices == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, slice_idx in enumerate(slice_indices):
        axes[idx].imshow(volume[slice_idx], cmap='gray')
        axes[idx].set_title(f'Slice {slice_idx}/{D-1}')
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(num_slices, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def compare_volumes(original: np.ndarray, processed: np.ndarray,
                    slice_idx: int = None, save_path: Optional[str] = None) -> None:
    """
    Compare original and processed volumes side by side
    
    Args:
        original: Original volume
        processed: Processed volume
        slice_idx: Slice index (middle if None)
        save_path: Path to save figure
    """
    if slice_idx is None:
        slice_idx = original.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original[slice_idx], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(processed[slice_idx], cmap='gray')
    axes[1].set_title('Processed')
    axes[1].axis('off')
    
    diff = np.abs(processed[slice_idx] - original[slice_idx])
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    
    plt.suptitle(f'Comparison - Slice {slice_idx}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def get_volume_stats(volume: np.ndarray, name: str = "Volume") -> dict:
    """
    Calculate and print volume statistics
    
    Args:
        volume: Input volume
        name: Volume name for display
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'shape': volume.shape,
        'dtype': volume.dtype,
        'min': float(volume.min()),
        'max': float(volume.max()),
        'mean': float(volume.mean()),
        'std': float(volume.std()),
        'size_mb': volume.nbytes / (1024**2)
    }
    
    print(f"\n{name} Statistics:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Data type: {stats['dtype']}")
    print(f"  Value range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    print(f"  Memory: {stats['size_mb']:.2f} MB")
    
    return stats

