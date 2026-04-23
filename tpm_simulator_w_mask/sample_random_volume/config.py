"""
Config for MICrONS volume download and stitch.
"""

# Where to write run outputs (default: current dir)
OUTPUT_ROOT_DIR = '/cis/home/rwang183/my_documents/BDD/SpineDetect-shengwen-branch/SpineDetect/tpm_simulator/data/download'
# Override log path when run from download.py pipeline (set by patch)
LOG_DIR = '/cis/home/rwang183/my_documents/BDD/SpineDetect-shengwen-branch/SpineDetect/tpm_simulator/pipelines/cloud/logs/download'
LOG_FILE = '/cis/home/rwang183/my_documents/BDD/SpineDetect-shengwen-branch/SpineDetect/tpm_simulator/pipelines/cloud/logs/download/download_20260411_134911.log'

# Segmentation MIP level (higher = coarser resolution)
MIP_LEVEL = 3

# Block and grid: each block is BLOCK_PX^3 voxels
BLOCK_PX = 128
# Neuron grid (em + svx)
NEURON_GRID_SIZE_X = 4
NEURON_GRID_SIZE_Y = 4
NEURON_GRID_SIZE_Z = 8
# Vessel grid (vas); same origin as neuron, larger extent
VESSEL_GRID_SIZE_X = 12
VESSEL_GRID_SIZE_Y = 12
VESSEL_GRID_SIZE_Z = 12

# Voxel origin for first block (at current MIP)
VOXEL_ORIGIN_X = 0
VOXEL_ORIGIN_Y = 0
VOXEL_ORIGIN_Z = 0

# Parallel workers for block download
DOWNLOAD_WORKERS = 8
# Block download timeout (seconds); exceed triggers retry. 180 = 3 min
BLOCK_DOWNLOAD_TIMEOUT = 180

# Parallel workers for stitching
STITCH_WORKERS = 4

# CAVE dataset and materialization
DATASET_NAME = "minnie65_public"
MATERIALIZATION_VERSION = 1300
# Fraction of found IDs to sample (neurons / vessels)
NEURON_SAMPLING_RATIO = 0.01725603108026936
VESSEL_SAMPLING_RATIO = 0.5596987104980888
# Set False to disable vessel mask. When True, uses pericyte+astrocyte (vessel-associated cells) from aibs_metamodel_celltypes_v661.
USE_VESSEL_MASK = True

# CloudVolume layer URLs
IMAGE_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em"
SEGMENTATION_URL = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300"
