"""
Config for MICrONS volume download and stitch.
"""

# Where to write run outputs (default: current dir)
OUTPUT_ROOT_DIR = "."

# Segmentation MIP level (higher = coarser resolution)
MIP_LEVEL = 6

# Block and grid: each block is BLOCK_PX^3 voxels; grid is GRID_SIZE^3 blocks
BLOCK_PX = 128
GRID_SIZE = 4

# Voxel origin for first block (at current MIP)
VOXEL_ORIGIN_X = 0
VOXEL_ORIGIN_Y = 0
VOXEL_ORIGIN_Z = 0

# Parallel workers for block download
DOWNLOAD_WORKERS = 8

# Parallel workers for stitching
STITCH_WORKERS = 4

# CAVE dataset and materialization
DATASET_NAME = "minnie65_public"
MATERIALIZATION_VERSION = 1300
# Fraction of found IDs to sample (neurons / vessels)
NEURON_SAMPLING_RATIO = 0.05
VESSEL_SAMPLING_RATIO = 0.5

# CloudVolume layer URLs
IMAGE_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em"
SEGMENTATION_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300"
