"""
Configuration file for Neural Field 3D Image Restoration
"""

import os

class Config:
    # ==================== Path Settings ====================
    # Training data directory
    TRAIN_DATA_DIR = r"D:\资料\JHU\课程\Biomedical Data Design I\SpineDetect\Test\Data\train"
    
    # Output directories
    OUTPUT_DIR = r"D:\资料\JHU\课程\Biomedical Data Design I\SpineDetect\Test\INF\outputs"
    CHECKPOINT_DIR = r"D:\资料\JHU\课程\Biomedical Data Design I\SpineDetect\Test\INF\checkpoints"
    LOG_DIR = r"D:\资料\JHU\课程\Biomedical Data Design I\SpineDetect\Test\INF\logs"
    
    # ==================== Model Architecture ====================
    # Coordinate encoder
    NUM_FREQS = 6  # Fourier feature frequencies
    
    # Local encoder
    LOCAL_BASE_CHANNELS = 16
    LOCAL_LATENT_DIM = 64
    
    # Global encoder
    GLOBAL_BASE_CHANNELS = 16
    GLOBAL_LATENT_DIM = 64
    GLOBAL_DOWNSAMPLE_FACTOR = 0.25  # Deprecated: use GLOBAL_TARGET_SHAPE instead
    GLOBAL_TARGET_SHAPE = (32, 32, 32)  # Fixed target shape for global context
    
    # Neural Field MLP
    MLP_HIDDEN_DIM = 128
    MLP_NUM_LAYERS = 4
    RESIDUAL_MODE = True  # Predict residual instead of direct value
    RESIDUAL_SCALE = 0.2  # Scale factor for bounded residual output
    
    # ==================== Training Settings ====================
    # Block sampling
    BLOCK_SIZE = 64  # 64x64x64 blocks
    NUM_VOXEL_SAMPLES = 8192  # K points per block per step
    FOREGROUND_RATIO = 0.75  # 75% foreground, 25% background
    FOREGROUND_THRESHOLD = 0.1  # Threshold to determine foreground
    
    # Loss weights (reduced to prevent over-brightening)
    WEIGHT_FOREGROUND = 1.5  # Changed from 4.0 to prevent global brightening
    WEIGHT_BACKGROUND = 1.0
    
    # Conservative regularization: penalize deviations from input
    USE_CONSERVATIVE_REG = True  # Add penalty for changing too much from input
    CONSERVATIVE_REG_WEIGHT = 0.05  # Weight for conservative regularization
    
    # Training hyperparameters
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    STEPS_PER_EPOCH = 100  # Number of block samples per epoch
    
    # ==================== Inference Settings ====================
    # Sliding window inference
    INFERENCE_BLOCK_SIZE = 64
    INFERENCE_OVERLAP = 16  # Overlap between blocks
    INFERENCE_CHUNK_SIZE = 8192  # Process voxels in chunks to save memory
    
    # Use Hann window for smooth blending
    USE_HANN_WINDOW = True
    WINDOW_ALPHA = 0.5
    WINDOW_MIN_VALUE = 1e-3
    
    # Conservative blending: mix prediction with input for safer results
    USE_CONSERVATIVE_BLEND = False  # Set True for extra safety
    CONSERVATIVE_BLEND_ALPHA = 0.7  # 0.0=input only, 1.0=prediction only, 0.7=mostly prediction
    
    # ==================== Hardware Settings ====================
    DEVICE = "cuda"  # or "cpu"
    NUM_WORKERS = 2  # Data loading workers
    
    # ==================== Data Augmentation ====================
    USE_AUGMENTATION = True
    FLIP_PROB = 0.5
    ROTATE_PROB = 0.5
    
    # Degradation simulation (for training)
    DEGRADATION_NOISE_STD = 0.03  # Reduced from 0.05 for gentler training
    
    # ==================== Logging Settings ====================
    LOG_INTERVAL = 10  # Log every N steps
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

