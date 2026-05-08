# Spine Detect

The purpose of this project is to train a deep learning model to automatically segement dendritic spines in 2-Photon microscopy images. To do this, we generate a large synthetic traing dataset from MICrONS that is 3 orders of magnitude larger than previously published methods.

## Project Setup

Note that the environment uses the `tensorflow[and-cuda]` Python package, which is only supported on Linux. If you're using Windows, you must use WSL2. If you're on Mac, update the requirement in `environment.yml` with the approprate package name (probably just `tensorflow`)

```bash
# Setup the virtual environment
conda env create --name spine-detect python=3.12.13

conda activate spine-detect

pip install -r ./2-Dataset-Generation/tpm_simulator_w_mask/requirements.txt
pip install tensorflow[and-cuda] keras natsort
```

Once your environment is established, make sure you're authenticated with the MICrONS platform by following the [instructions on their website](https://tutorial.microns-explorer.org/quickstart_notebooks/01-caveclient-setup.html).
