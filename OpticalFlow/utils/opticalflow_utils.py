"""
Optical Flow Utilities for 3D Volume Analysis
Implements optical flow computation between consecutive slices
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm


def compute_optical_flow_3d(volume: np.ndarray, 
                            method: str = 'tvl1',
                            **kwargs) -> np.ndarray:
    """
    Compute optical flow between consecutive slices in 3D volume
    
    Args:
        volume: 3D volume (Z, H, W) normalized to [0, 1]
        method: Optical flow method ('tvl1', 'farneback', 'deepflow')
        **kwargs: Additional parameters for the optical flow algorithm
        
    Returns:
        flow_volume: 4D array (Z-1, H, W, 2) containing (u, v) flow vectors
    """
    Z, H, W = volume.shape
    flow_volume = np.zeros((Z-1, H, W, 2), dtype=np.float32)
    
    print(f"Computing {method.upper()} optical flow for {Z-1} slice pairs...")
    
    for z in tqdm(range(Z-1), desc="Computing flow"):
        # Get consecutive slices
        slice1 = prepare_slice_for_flow(volume[z])
        slice2 = prepare_slice_for_flow(volume[z+1])
        
        # Compute optical flow
        if method == 'tvl1':
            flow = compute_tvl1_flow(slice1, slice2, **kwargs)
        elif method == 'farneback':
            flow = compute_farneback_flow(slice1, slice2, **kwargs)
        elif method == 'deepflow':
            flow = compute_deepflow(slice1, slice2, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        flow_volume[z] = flow
    
    return flow_volume


def prepare_slice_for_flow(slice_img: np.ndarray) -> np.ndarray:
    """
    Prepare slice for optical flow computation
    
    Args:
        slice_img: 2D slice in [0, 1] range
        
    Returns:
        Prepared slice as uint8
    """
    # Ensure [0, 1] range
    slice_img = np.clip(slice_img, 0, 1)
    # Convert to uint8 [0, 255]
    slice_uint8 = (slice_img * 255).astype(np.uint8)
    return slice_uint8


def compute_tvl1_flow(slice1: np.ndarray, slice2: np.ndarray,
                      tau: float = 0.25,
                      lambda_: float = 0.15,
                      theta: float = 0.3,
                      nscales: int = 5,
                      warps: int = 5,
                      epsilon: float = 0.01,
                      inner_iterations: int = 30,
                      outer_iterations: int = 10,
                      scale_step: float = 0.8,
                      gamma: float = 0.0,
                      median_filtering: int = 5,
                      use_initial_flow: bool = False) -> np.ndarray:
    """
    Compute Dual TV-L1 optical flow (high precision)
    
    Based on "Dual TV L1" Optical Flow Algorithm
    Reference: Zach et al. "A Duality Based Approach for Realtime TV-L1 Optical Flow"
    
    Args:
        slice1, slice2: Input slices (uint8)
        tau: Time step
        lambda_: Weight for data term
        theta: Weight for warping
        nscales: Number of pyramid scales
        warps: Number of warpings per scale
        epsilon: Stopping criterion
        inner_iterations: Inner fixed-point iterations
        outer_iterations: Outer fixed-point iterations
        scale_step: Pyramid scale step
        gamma: Gradient constancy weight
        median_filtering: Median filter radius
        use_initial_flow: Use initial flow
        
    Returns:
        flow: Optical flow (H, W, 2)
    """
    # Create DualTVL1 optical flow object
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
        tau=tau,
        lambda_=lambda_,
        theta=theta,
        nscales=nscales,
        warps=warps,
        epsilon=epsilon,
        innnerIterations=inner_iterations,
        outerIterations=outer_iterations,
        scaleStep=scale_step,
        gamma=gamma,
        medianFiltering=median_filtering,
        useInitialFlow=use_initial_flow
    )
    
    # Compute flow
    flow = tvl1.calc(slice1, slice2, None)
    
    return flow


def compute_farneback_flow(slice1: np.ndarray, slice2: np.ndarray,
                           pyr_scale: float = 0.5,
                           levels: int = 5,
                           winsize: int = 15,
                           iterations: int = 3,
                           poly_n: int = 5,
                           poly_sigma: float = 1.2,
                           flags: int = 0) -> np.ndarray:
    """
    Compute Farneback optical flow (classic method)
    
    Args:
        slice1, slice2: Input slices (uint8)
        pyr_scale: Pyramid scale
        levels: Number of pyramid levels
        winsize: Window size
        iterations: Number of iterations
        poly_n: Polynomial expansion neighborhood size
        poly_sigma: Gaussian sigma for polynomial expansion
        flags: Operation flags
        
    Returns:
        flow: Optical flow (H, W, 2)
    """
    flow = cv2.calcOpticalFlowFarneback(
        slice1, slice2, None,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags
    )
    
    return flow


def compute_deepflow(slice1: np.ndarray, slice2: np.ndarray,
                     **kwargs) -> np.ndarray:
    """
    Compute DeepFlow optical flow (requires opencv-contrib)
    
    Args:
        slice1, slice2: Input slices (uint8)
        **kwargs: Additional parameters
        
    Returns:
        flow: Optical flow (H, W, 2)
    """
    try:
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        flow = deepflow.calc(slice1, slice2, None)
        return flow
    except AttributeError:
        print("DeepFlow not available, falling back to TV-L1")
        return compute_tvl1_flow(slice1, slice2, **kwargs)


def compute_flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute magnitude of optical flow vectors
    
    Args:
        flow: Optical flow (H, W, 2) or (Z, H, W, 2)
        
    Returns:
        magnitude: Flow magnitude
    """
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.sqrt(u**2 + v**2)
    return magnitude


def compute_flow_angle(flow: np.ndarray) -> np.ndarray:
    """
    Compute angle of optical flow vectors
    
    Args:
        flow: Optical flow (H, W, 2) or (Z, H, W, 2)
        
    Returns:
        angle: Flow angle in radians [-pi, pi]
    """
    u = flow[..., 0]
    v = flow[..., 1]
    angle = np.arctan2(v, u)
    return angle


def flow_to_color(flow: np.ndarray, max_flow: Optional[float] = None) -> np.ndarray:
    """
    Convert optical flow to color visualization (HSV -> RGB)
    
    Args:
        flow: Optical flow (H, W, 2)
        max_flow: Maximum flow for normalization (auto if None)
        
    Returns:
        flow_color: RGB image (H, W, 3) in [0, 255] uint8
    """
    H, W = flow.shape[:2]
    
    # Compute magnitude and angle
    mag = compute_flow_magnitude(flow)
    ang = compute_flow_angle(flow)
    
    # Normalize magnitude
    if max_flow is None:
        max_flow = mag.max()
    mag_norm = np.clip(mag / (max_flow + 1e-10), 0, 1)
    
    # Create HSV image
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue: angle
    hsv[..., 1] = 255  # Saturation: full
    hsv[..., 2] = (mag_norm * 255).astype(np.uint8)  # Value: magnitude
    
    # Convert to RGB
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flow_color


def filter_flow_by_magnitude(flow: np.ndarray, threshold: float) -> np.ndarray:
    """
    Filter optical flow by magnitude threshold
    
    Args:
        flow: Optical flow (H, W, 2) or (Z, H, W, 2)
        threshold: Magnitude threshold
        
    Returns:
        filtered_flow: Flow with small magnitudes set to zero
    """
    mag = compute_flow_magnitude(flow)
    mask = mag < threshold
    
    filtered_flow = flow.copy()
    filtered_flow[mask] = 0
    
    return filtered_flow


def smooth_flow_temporal(flow_volume: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Temporally smooth flow volume with moving average
    
    Args:
        flow_volume: 4D flow volume (Z, H, W, 2)
        window: Temporal window size
        
    Returns:
        smoothed_flow: Temporally smoothed flow
    """
    Z = flow_volume.shape[0]
    smoothed_flow = np.zeros_like(flow_volume)
    
    for z in range(Z):
        z_start = max(0, z - window//2)
        z_end = min(Z, z + window//2 + 1)
        smoothed_flow[z] = flow_volume[z_start:z_end].mean(axis=0)
    
    return smoothed_flow

