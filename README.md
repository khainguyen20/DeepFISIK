# Deep-FISIK

Deep-FISIK is a graph neural network framework for estimating molecular interaction parameters from single-molecule imaging data. The repository contains classification models for oligomer-state prediction and regression models for estimating interaction kinetics from simulated and experimental data.

---

## Features

* Maximum oligomer system classification (monomer, dimer, trimer, tetramer)
* SMI parameter estimation
* Dimer, trimer, and tetramer regression models
* Training and inference pipelines
* Pretrained checkpoints
* Support for both labeled and unlabeled datasets

---

## Installation

### Clone the repository

```bash
git clone https://github.com/khainguyen20/DeepFISIK.git
cd Deep-FISIK
```

### Create the conda environment

```bash
conda env create -f environment.yml
conda activate deepfisik
```

### Install Deep-FISIK

```bash
pip install -r requirements.txt 
```

Verify the installation:

```bash
python -c "import deepfisik"
```

---

## Repository Structure

```text
Deep-FISIK/
├── Datasets/
│   ├── Images/
│   │   ├── MonomerDataset/
│   │   ├── DimerDataset/
│   │   ├── TrimerDataset/
│   │   └── TetramerDataset/
│   │
│   └── PureSimulations/
│       ├── MonomerDataset/
│       ├── DimerDataset/
│       ├── TrimerDataset/
│       └── TetramerDataset/
│
├── checkpoints/
├── scripts/
├── src/
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

---

## Datasets

Place datasets in the `Datasets/` directory.

### Classification datasets

```text
Datasets/
└── Images/
    ├── MonomerDataset/
    ├── DimerDataset/
    ├── TrimerDataset/
    └── TetramerDataset/
```

### Interaction datasets

```text
Datasets/
└── PureSimulations/
    ├── MonomerDataset/
    ├── DimerDataset/
    ├── TrimerDataset/
    └── TetramerDataset/
```

---

## Training

### Classification

```bash
python scripts/train_classification.py
```

### Dimer regression

```bash
python scripts/train_dimers.py
```

### Trimer regression

```bash
python scripts/train_trimers.py
```

### Tetramer regression

```bash
python scripts/train_tetramers.py
```

Custom datasets can be specified using:

```bash
python scripts/train_dimers.py \
    --dataset-roots path/to/dataset1 path/to/dataset2
```

Multiple datasets are automatically concatenated.

---

## Inference

### Classification

```bash
python scripts/predict_classification.py
```

### Dimer

```bash
python scripts/predict_dimers.py
```

### Trimer

```bash
python scripts/predict_trimers.py
```

### Tetramer

```bash
python scripts/predict_tetramers.py
```

### Using a custom checkpoint

```bash
python scripts/predict_tetramers.py \
    --checkpoint checkpoints/my_model.pt
```

---

## Pretrained Models

Pretrained models are provided in:

```text
models/
├── Images/
└── PureSimulations/
```

These checkpoints can be used directly for inference.

---

## Outputs

Training runs generate:

```text
runs/
├── parameters/
├── results/
├── checkpoints/
└── finalModel/
```

---

## Citation

If you use Deep-FISIK in your research, please cite:

```text
Citation information will be added upon publication.
```

---

## License

Add the appropriate license information for your project here.




