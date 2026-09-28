# StresSense Training Pipeline

This directory contains the training pipeline for the public StresSense dataset.

## Project structure

```text
stresssense/
├── README.md
├── src/
│   ├── config.py            # Paths, sensor columns, and default settings
│   ├── data_utils.py        # Dataset loading, features, and window creation
│   ├── inspect_dataset.py   # Dataset and participant summaries
│   ├── preprocess.py        # Cross-user splits and normalization
│   ├── train.py             # CNN training, evaluation, and plots
│   └── quantize.py          # TFLite conversion and evaluation
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training.ipynb
│   └── 03_quantization.ipynb
├── data/
│   ├── raw/                # Downloaded StresSense.csv
│   └── processed/          # Generated arrays and preprocessing metadata
├── models/                 # Trained Keras and TFLite models
└── results/                # Metrics, classification report, and plots
```

The notebooks import the implementation from `src/`; keep both folders when
sharing the notebook workflow. Paths in `src/config.py` resolve relative to
the `stresssense/` directory, so outputs are not written inside `src/`.

## Environment

The existing development environment is named `tinyml2` and uses Python 3.10
with TensorFlow 2.13. The pipeline also uses NumPy, pandas, scikit-learn,
Matplotlib, and seaborn. Jupyter Notebook and IPython are needed for the
notebook workflow. If your environment has a different name, replace
`tinyml2` in the commands below.

## Dataset

- Official dataset: <https://data.mendeley.com/datasets/2dn3hpbm5m/1>
- DOI: `10.17632/2dn3hpbm5m.1`
- File expected by the scripts: `data/raw/StresSense.csv`
- 495,446 samples from 40 participants at 50 Hz
- Activities: smoking, eating, nail biting, face touching, and staying still
- Sensors: wrist accelerometer, gyroscope, and magnetometer

Download the dataset manually and place the CSV here:

```text
stresssense/data/raw/StresSense.csv
```

## Jupyter notebooks

From the repository root, start Jupyter in the `stresssense` directory:

```bash
conda activate tinyml2
cd stresssense
jupyter notebook
```

Run the notebooks in this order:

1. `notebooks/01_preprocessing.ipynb`
2. `notebooks/02_training.ipynb`
3. `notebooks/03_quantization.ipynb`

The notebooks call the Python pipeline files in `src/` and display the
saved metrics and plots. Data, models, and results remain in their existing
folders under `stresssense/`. Restart the notebook kernel if it was open
before the scripts were moved into `src/`.

The first notebook calls `inspect_dataset.py` and `preprocess.py`, the second
calls `train.py`, and the third calls `quantize.py`. Running these notebooks
executes the pipeline; it does not merely display the saved outputs.

## Command-line alternative

From the repository root, activate the environment and enter `stresssense/`.
If you are already in that directory, skip the `cd` command:

```bash
conda activate tinyml2
cd stresssense
```

### 1. Inspect the dataset

```bash
python src/inspect_dataset.py
```

This prints the detected columns, activity labels, user IDs, missing values,
and users that contain every activity.

### 2. Preprocess

Deployment-compatible accelerometer and gyroscope input:

```bash
python src/preprocess.py --sensor-profile ag --test-user auto
```

Full StresSense accelerometer, gyroscope, and magnetometer input:

```bash
python src/preprocess.py --sensor-profile agm --test-user auto
```

Important options:

```text
--window-size 250     5-second windows at 50 Hz
--overlap 0.8         80% overlap, following the dataset paper
--test-user auto      hold out one user containing all five activities
--validation-user auto hold out a second user for model selection
--add-magnitude       add sensor magnitude features
```

Magnitude features are enabled by default. Use `--no-add-magnitude` to
disable them. The default `ag` profile therefore produces eight features:
six sensor axes and two magnitudes.

Normalization statistics are calculated from training users only. Windows
never cross user or activity boundaries.

### 3. Train

Training starts only when this command is run:

```bash
python src/train.py
```

The model is a quantization-friendly 1D CNN without BatchNorm. Outputs:

```text
models/stresssense_model.h5
models/training_info.pkl
results/metrics.json
results/classification_report.txt
results/training_history.png
results/confusion_matrix.png
```

### 4. Quantize

```bash
python src/quantize.py
```

This creates and evaluates:

- Full INT8 model for TensorFlow Lite Micro
- Hybrid INT8-weight model with float input/output

The converted models are saved to `models/stresssense_full_int8.tflite` and
`models/stresssense_hybrid.tflite`. The quantization report is saved to
`results/quantization_metrics.json`.

## Generated files and version control

The repository's `.gitignore` excludes the raw dataset and generated files
inside `data/processed/`, `models/`, and `results/`. The `.gitkeep` files retain
these directories in Git. After cloning, download the dataset and run
preprocessing, training, and quantization in order to regenerate the outputs.

IDE settings (`.idea/`), Python caches (`__pycache__/`), notebook checkpoints
(`.ipynb_checkpoints/`), macOS metadata (`.DS_Store`), and temporary files
(`tmp/`) are also ignored and are not needed in a submission archive. Include
models and results separately if they are required submission deliverables.

## Fair evaluation

The split is cross-user: one complete participant is held out for validation
and another for final testing. Their windows are excluded from training and
normalization. The test user is never used for model selection or calibration.
