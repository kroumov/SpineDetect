# Synthetic data generation pipeline
This pipeline requires `python==3.12`

To setup a virtual environment:
```
python -m venv microns_env
source microns_env/bin/activate
pip install -r requirements.txt
```
To run the pipeline:
```
bash pipelines/cloud/run_pipeline.sh
```
Final output will be available in `data/train`
