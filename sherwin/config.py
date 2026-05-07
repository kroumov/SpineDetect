"""
Configuration for MICrONS random volume sampling and accumulation.
"""

# Output root directory for downloaded blocks and merged volumes
OUTPUT_ROOT_DIR = "."

# Target voxel resolution (nm). Powers of 2 work well for block_reduce.
RES_NM_X = 128
RES_NM_Y = 128
RES_NM_Z = 128

# Block size in output voxels. Each block yields one em_*.npy and svx_*.npy file.
# BLOCK_PX_Z=40 with RES_NM_Z=128 yields 40 slices at 128nm, matching common EM slab thickness.
BLOCK_PX_X = 64
BLOCK_PX_Y = 64
BLOCK_PX_Z = 40

# Grid extent: number of blocks along each axis, relative to center.
# X_BLOCKS_NEG=4, X_BLOCKS_POS=4 -> X indices -4..3, 8 blocks total.
X_BLOCKS_NEG = 4
X_BLOCKS_POS = 4
Y_BLOCKS_NEG = 4
Y_BLOCKS_POS = 4
Z_BLOCKS_NEG = 4
Z_BLOCKS_POS = 4

# Offset added to nucleus position (nm) before computing ROI center
OFFSET_NM_X = 0
OFFSET_NM_Y = 0
OFFSET_NM_Z = 0

# Fraction of neurons in the ROI to include in the mask (2P-like sparse labeling)
NEURON_SAMPLING_RATIO = 0.01

# MIP level used when querying segmentation for preview and neuron selection.
# Higher MIP = faster but coarser.
SEGMENTATION_MIP_LEVEL = 3

# MICrONS dataset and materialization
DATASET_NAME = "minnie65_public"
MATERIALIZATION_VERSION = 1300

# CloudVolume precomputed URLs
IMAGE_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em"
SEGMENTATION_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300"

# Number of parallel download workers
PARALLEL_JOBS = 8