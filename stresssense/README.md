# StresSense Training Pipeline

This branch contains an isolated pipeline for the public StresSense dataset.
It does not modify the ADAM-sense notebooks or models.

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

Start Jupyter from the `stresssense` directory:

```bash
conda activate tinyml2
cd /Users/jaloliddinabdullaev/Projects/activity-recognition-training/stresssense
jupyter notebook
```

Run the notebooks in this order:

1. `notebooks/01_preprocessing.ipynb`
2. `notebooks/02_training.ipynb`
3. `notebooks/03_quantization.ipynb`

The notebooks call the Python pipeline files in this directory and display the
saved metrics and plots.

## Command-line alternative

Use the existing TensorFlow environment:

```bash
conda activate tinyml2
cd stresssense
```

### 1. Inspect the dataset

```bash
python inspect_dataset.py
```

This prints the detected columns, activity labels, user IDs, missing values,
and users that contain every activity.

### 2. Preprocess

Deployment-compatible accelerometer and gyroscope input:

```bash
python preprocess.py --sensor-profile ag --test-user auto
```

Full StresSense accelerometer, gyroscope, and magnetometer input:

```bash
python preprocess.py --sensor-profile agm --test-user auto
```

Important options:

```text
--window-size 250     5-second windows at 50 Hz
--overlap 0.8         80% overlap, following the dataset paper
--test-user auto      hold out one user containing all five activities
--validation-user auto hold out a second user for model selection
--add-magnitude       add sensor magnitude features
```

Normalization statistics are calculated from training users only. Windows
never cross user or activity boundaries.

### 3. Train

Training starts only when this command is run:

```bash
python train.py
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
python quantize.py
```

This creates and evaluates:

- Full INT8 model for TensorFlow Lite Micro
- Hybrid INT8-weight model with float input/output

The quantization report is saved to `results/quantization_metrics.json`.

## Fair evaluation

The split is cross-user: one complete participant is held out for validation
and another for final testing. Their windows are excluded from training and
normalization. The test user is never used for model selection or calibration.
