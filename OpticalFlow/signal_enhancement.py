"""
Task 1: Optical Flow-Based Signal Enhancement

Uses optical flow to enhance dendrite and spine signals while suppressing background:

Main Goal: ENHANCE dendrites and spines (signal regions with consistent flow)
Secondary Goal: Suppress background (regions without consistent flow)

The algorithm works by:
1. Computing optical flow between consecutive Z-slices
2. Identifying regions with consistent flow (dendrites/spines moving through Z)
3. STRONGLY ENHANCING these signal regions
4. Optionally suppressing background regions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
from typing import Tuple, Optional

from utils.io_utils import (
    load_tif_volume, save_tif_volume, normalize_volume,
    visualize_slices, compare_volumes, get_volume_stats
)
from utils.opticalflow_utils import (
    compute_optical_flow_3d, compute_flow_magnitude,
    smooth_flow_temporal
)


class SignalEnhancer:
    """
    Optical flow-based signal enhancement for dendrites and spines
    
    Focus: ENHANCE signal regions, optionally suppress background
    """
    
    def __init__(self,
                 flow_method: str = 'tvl1',
                 signal_threshold: float = 0.3,
                 signal_percentile: Optional[float] = None,
                 consistency_window: int = 5,
                 enhancement_strength: float = 3.0,
                 enhancement_power: float = 1.5,
                 background_suppression: float = 0.7,
                 adaptive_enhancement: bool = True,
                 edge_preserve_sigma: float = 0.1,
                 use_roi_clahe: bool = True):
        """
        Initialize signal enhancer
        
        Args:
            flow_method: Optical flow method ('tvl1' for precision)
            signal_threshold: Minimum flow to consider as signal (absolute value)
            signal_percentile: If provided, use percentile instead of absolute threshold (e.g., 90 = top 10%)
            consistency_window: Window for checking flow consistency
            enhancement_strength: Signal enhancement multiplier (>1, recommend 2-5)
            enhancement_power: Power for non-linear enhancement (>1)
            background_suppression: Background suppression factor (0-1, closer to 1 = less suppression)
            adaptive_enhancement: Use adaptive enhancement based on flow strength
            edge_preserve_sigma: Sigma for edge-preserving smoothing
            use_roi_clahe: Apply CLAHE only on signal ROI (more efficient)
        """
        self.flow_method = flow_method
        self.signal_threshold = signal_threshold
        self.signal_percentile = signal_percentile
        self.consistency_window = consistency_window
        self.enhancement_strength = enhancement_strength
        self.enhancement_power = enhancement_power
        self.background_suppression = background_suppression
        self.adaptive_enhancement = adaptive_enhancement
        self.edge_preserve_sigma = edge_preserve_sigma
        self.use_roi_clahe = use_roi_clahe
        
    def process(self, volume: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Enhance signal in 3D volume
        
        Args:
            volume: Input volume (Z, H, W) normalized to [0, 1]
            verbose: Print progress
            
        Returns:
            enhanced_volume: Signal-enhanced volume
            info: Processing information dictionary
        """
        if verbose:
            print("\n" + "="*60)
            print("Optical Flow-Based Signal Enhancement")
            print("Focus: ENHANCE dendrites and spines")
            print("="*60)
            get_volume_stats(volume, "Input Volume")
        
        # Step 1: Compute optical flow
        if verbose:
            print("\n[Step 1] Computing optical flow (detecting signal movement)...")
        flow_volume = compute_optical_flow_3d(volume, method=self.flow_method)
        
        # Step 2: Compute signal strength map
        if verbose:
            print("\n[Step 2] Computing signal strength map...")
        signal_strength = self._compute_signal_strength(flow_volume)
        
        # Step 3: Identify signal vs background regions
        if verbose:
            print("\n[Step 3] Identifying signal regions...")
        
        # Use percentile threshold if provided
        if self.signal_percentile is not None:
            threshold = np.percentile(signal_strength, self.signal_percentile)
            if verbose:
                print(f"  Using percentile threshold: {self.signal_percentile}% = {threshold:.4f}")
        else:
            threshold = self.signal_threshold
            if verbose:
                print(f"  Using absolute threshold: {threshold:.4f}")
        
        signal_mask = signal_strength > threshold
        
        # Step 4: ENHANCE signal regions
        if verbose:
            print("\n[Step 4] ENHANCING dendrite and spine signals...")
        enhanced_volume = self._enhance_signal_regions(
            volume, signal_strength, signal_mask
        )
        
        # Statistics
        signal_ratio = signal_mask.sum() / signal_mask.size
        enhancement_stats = {
            'mean_original': float(volume.mean()),
            'mean_enhanced': float(enhanced_volume.mean()),
            'max_original': float(volume.max()),
            'max_enhanced': float(enhanced_volume.max()),
            'signal_mean_original': float(volume[signal_mask].mean()) if signal_mask.any() else 0,
            'signal_mean_enhanced': float(enhanced_volume[signal_mask].mean()) if signal_mask.any() else 0,
        }
        
        if verbose:
            print(f"\nSignal coverage: {signal_ratio:.2%}")
            print(
                f"Mean intensity: "
                f"{enhancement_stats['mean_original']:.4f} → {enhancement_stats['mean_enhanced']:.4f}"
            )
            print(
                f"Signal region mean: "
                f"{enhancement_stats['signal_mean_original']:.4f} → {enhancement_stats['signal_mean_enhanced']:.4f}"
            )
            print(
                f"Enhancement factor: "
                f"{enhancement_stats['signal_mean_enhanced'] / (enhancement_stats['signal_mean_original'] + 1e-10):.2f}x"
            )
            get_volume_stats(enhanced_volume, "Enhanced Volume")
        
        info = {
            'flow_volume': flow_volume,
            'signal_strength': signal_strength,
            'signal_mask': signal_mask,
            'signal_ratio': signal_ratio,
            'enhancement_stats': enhancement_stats
        }
        
        return enhanced_volume, info
    
    def _compute_signal_strength(self, flow_volume: np.ndarray) -> np.ndarray:
        """
        Compute signal strength based on flow consistency
        
        Strong consistent flow = dendrite/spine signal
        Weak/inconsistent flow = background
        
        Args:
            flow_volume: 4D flow (Z-1, H, W, 2)
            
        Returns:
            signal_strength: 3D map (Z, H, W) in [0, 1]
        """
        Z_flow, H, W, _ = flow_volume.shape
        Z = Z_flow + 1
        
        # Compute flow magnitude
        flow_mag = compute_flow_magnitude(flow_volume)
        
        # Smooth flow temporally to get consistent signal
        flow_mag_smooth = smooth_flow_temporal(
            flow_mag[..., np.newaxis], 
            window=self.consistency_window
        ).squeeze()
        
        # Extend to full Z dimension
        signal_strength = np.zeros((Z, H, W), dtype=np.float32)
        signal_strength[:-1] = flow_mag_smooth
        signal_strength[-1] = flow_mag_smooth[-1]
        
        # Compute consistency score (low temporal variation = consistent signal)
        for z in range(Z):
            z_start = max(0, z - self.consistency_window//2)
            z_end = min(Z, z + self.consistency_window//2 + 1)
            
            local_signals = signal_strength[z_start:z_end]
            mean_signal = local_signals.mean(axis=0)
            std_signal = local_signals.std(axis=0)
            
            # Consistency: high mean, low std = strong consistent signal
            consistency = mean_signal / (std_signal + mean_signal * 0.1 + 1e-6)
            consistency = np.clip(consistency, 0, 1)
            
            # Combine magnitude and consistency
            signal_strength[z] = mean_signal * consistency
        
        # Normalize to [0, 1]
        signal_max = np.percentile(signal_strength, 99)
        if signal_max > 0:
            signal_strength = signal_strength / signal_max
        signal_strength = np.clip(signal_strength, 0, 1)
        
        return signal_strength
    
    def _enhance_signal_regions(self, volume: np.ndarray, 
                                 signal_strength: np.ndarray,
                                 signal_mask: np.ndarray) -> np.ndarray:
        """
        Enhance signal regions (dendrites and spines)
        
        Main focus: STRONG enhancement of signal regions
        Secondary: Optional background suppression
        
        Args:
            volume: Original volume
            signal_strength: Signal strength map [0, 1]
            signal_mask: Binary signal mask
            
        Returns:
            enhanced_volume: Enhanced volume
        """
        enhanced = volume.copy()
        
        # Method 1: Adaptive enhancement based on signal strength
        if self.adaptive_enhancement:
            # Non-linear enhancement: stronger signals get more boost
            enhancement_map = 1.0 + (self.enhancement_strength - 1.0) * \
                              np.power(signal_strength, self.enhancement_power)
            
            # Apply enhancement
            enhanced = enhanced * enhancement_map
            
        else:
            # Simple enhancement: uniform boost to signal regions
            enhancement_map = np.ones_like(volume)
            enhancement_map[signal_mask] = self.enhancement_strength
            enhanced = enhanced * enhancement_map
        
        # Method 2: Background suppression (optional, less aggressive)
        if self.background_suppression < 1.0:
            background_mask = ~signal_mask
            background_factor = np.ones_like(volume)
            background_factor[background_mask] = self.background_suppression
            enhanced = enhanced * background_factor
        
        # Method 3: Local contrast enhancement on signal regions
        Z = volume.shape[0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        for z in range(Z):
            if not signal_mask[z].any():
                continue
            
            if self.use_roi_clahe:
                # Only apply CLAHE on ROI bounding box (more efficient, less background impact)
                ys, xs = np.where(signal_mask[z])
                if len(ys) == 0:
                    continue
                
                y0, y1 = max(0, ys.min() - 5), min(enhanced.shape[1], ys.max() + 5)
                x0, x1 = max(0, xs.min() - 5), min(enhanced.shape[2], xs.max() + 5)
                
                roi = enhanced[z, y0:y1, x0:x1]
                roi_uint16 = (roi * 65535).astype(np.uint16)
                roi_enhanced = clahe.apply(roi_uint16).astype(np.float32) / 65535
                
                # Blend within ROI based on signal strength
                w = signal_strength[z, y0:y1, x0:x1]
                enhanced[z, y0:y1, x0:x1] = roi * (1 - w * 0.3) + roi_enhanced * (w * 0.3)
            else:
                # Apply CLAHE to whole slice (original behavior)
                slice_uint16 = (enhanced[z] * 65535).astype(np.uint16)
                slice_enhanced = clahe.apply(slice_uint16).astype(np.float32) / 65535
                
                # Blend with original based on signal strength
                blend_weight = signal_strength[z]
                enhanced[z] = enhanced[z] * (1 - blend_weight * 0.3) + \
                             slice_enhanced * (blend_weight * 0.3)
        
        # Method 4: Edge-preserving smoothing to reduce artifacts
        if self.edge_preserve_sigma > 0:
            for z in range(Z):
                enhanced[z] = cv2.bilateralFilter(
                    enhanced[z].astype(np.float32),
                    d=5,
                    sigmaColor=self.edge_preserve_sigma,
                    sigmaSpace=5
                )
        
        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced


def enhance_signal(input_path: str, output_path: str,
                   flow_method: str = 'tvl1',
                   signal_threshold: float = 0.3,
                   signal_percentile: Optional[float] = None,
                   consistency_window: int = 5,
                   enhancement_strength: float = 3.0,
                   enhancement_power: float = 1.5,
                   background_suppression: float = 0.7,
                   adaptive: bool = True,
                   edge_sigma: float = 0.1,
                   use_roi_clahe: bool = True,
                   visualize: bool = True) -> None:
    """
    Main function to enhance signal in 3D volume
    
    Args:
        input_path: Input TIF file path
        output_path: Output TIF file path
        flow_method: Optical flow method ('tvl1' recommended)
        signal_threshold: Minimum signal strength (0-1)
        consistency_window: Window for consistency check
        enhancement_strength: Enhancement multiplier (recommend 2-5)
        enhancement_power: Non-linear enhancement power (>1)
        background_suppression: Background factor (0-1, closer to 1 = less suppression)
        adaptive: Use adaptive enhancement
        edge_sigma: Edge-preserving smoothing sigma
        visualize: Show visualizations
    """
    # Load volume
    print(f"Loading volume from: {input_path}")
    volume = load_tif_volume(input_path)
    volume_norm = normalize_volume(volume)
    
    # Initialize enhancer
    enhancer = SignalEnhancer(
        flow_method=flow_method,
        signal_threshold=signal_threshold,
        signal_percentile=signal_percentile,
        consistency_window=consistency_window,
        enhancement_strength=enhancement_strength,
        enhancement_power=enhancement_power,
        background_suppression=background_suppression,
        adaptive_enhancement=adaptive,
        edge_preserve_sigma=edge_sigma,
        use_roi_clahe=use_roi_clahe
    )
    
    # Process
    enhanced_volume, info = enhancer.process(volume_norm, verbose=True)
    
    # Save result
    save_tif_volume(enhanced_volume, output_path, bit_depth=16)
    
    # Save signal strength map
    signal_map_path = str(Path(output_path).with_name(
        Path(output_path).stem + "_signal_map.tif"
    ))
    save_tif_volume(info['signal_strength'], signal_map_path, bit_depth=16)
    
    # Visualizations
    if visualize:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compare original vs enhanced
        mid_slice = volume.shape[0] // 2
        compare_volumes(
            volume_norm, enhanced_volume, mid_slice,
            save_path=output_dir / "signal_enhancement_comparison.png"
        )
        
        # Visualize signal strength map
        visualize_slices(
            info['signal_strength'],
            title="Signal Strength Map (Flow Consistency)",
            save_path=output_dir / "signal_strength_map.png"
        )
        
        # Visualize signal mask
        visualize_slices(
            info['signal_mask'].astype(float),
            title="Signal Regions (Dendrites & Spines)",
            save_path=output_dir / "signal_mask.png"
        )
        
        # Enhancement factor visualization
        enhancement_factor = np.zeros_like(enhanced_volume)
        non_zero = volume_norm > 1e-6
        enhancement_factor[non_zero] = enhanced_volume[non_zero] / volume_norm[non_zero]
        visualize_slices(
            np.clip(enhancement_factor, 0, 5),
            title="Enhancement Factor Map",
            save_path=output_dir / "enhancement_factor.png"
        )
    
    print("\n" + "="*60)
    print("Signal Enhancement Complete!")
    print(f"Enhanced volume saved to: {output_path}")
    print(f"Signal map saved to: {signal_map_path}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optical Flow-Based Signal Enhancement for Dendrites and Spines"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input TIF file path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output TIF file path")
    parser.add_argument("--method", type=str, default="tvl1",
                        choices=["tvl1", "farneback"],
                        help="Optical flow method (tvl1 recommended for precision)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Signal threshold (0-1, lower = more sensitive)")
    parser.add_argument("--percentile", type=float, default=None,
                        help="Use percentile threshold instead (e.g., 90 = top 10%%)")
    parser.add_argument("--window", type=int, default=5,
                        help="Consistency window size")
    parser.add_argument("--enhance", type=float, default=3.0,
                        help="Enhancement strength (2-5 recommended)")
    parser.add_argument("--power", type=float, default=1.5,
                        help="Enhancement power for non-linearity (>1)")
    parser.add_argument("--bg-suppress", type=float, default=0.7,
                        help="Background suppression (0-1, 1=no suppression)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive enhancement")
    parser.add_argument("--edge-sigma", type=float, default=0.1,
                        help="Edge-preserving smoothing sigma")
    parser.add_argument("--no-roi-clahe", action="store_true",
                        help="Apply CLAHE to whole slice instead of ROI")
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable visualizations")
    
    args = parser.parse_args()
    
    enhance_signal(
        input_path=args.input,
        output_path=args.output,
        flow_method=args.method,
        signal_threshold=args.threshold,
        signal_percentile=args.percentile,
        consistency_window=args.window,
        enhancement_strength=args.enhance,
        enhancement_power=args.power,
        background_suppression=args.bg_suppress,
        adaptive=not args.no_adaptive,
        edge_sigma=args.edge_sigma,
        use_roi_clahe=not args.no_roi_clahe,
        visualize=not args.no_vis
    )

