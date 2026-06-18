# Datasets

## Simulation Generation

The synthetic datasets used to train and evaluate Deep-FISIK are generated in a framework described in:

**Guerrero, J., Malik, Z., Bilal, F., Jana, S., Dasgupta, A., and Jaqaman, K.**

*Inference of VEGFR2 dimerization kinetics on the cell surface by integrating single-molecule imaging and mathematical modeling* (bioRxiv, 2025).

The framework combines:

* stochastic simulation of receptor diffusion and interactions
* synthetic single-molecule imaging trajectories
* parameter inference from single-molecule imaging data

In Deep-FISIK, simulated trajectories are exported as CSV files and converted into graph datasets using the graph-generation utilities contained in:

```text
src/deepfisik/data/
```

The resulting graphs are stored as PyTorch Geometric `.pt` files and used for both classification and interaction-parameter inference.

For complete details regarding the biological system, simulation assumptions, stochastic modeling framework, and kinetic inference methodology, please refer to the original publication:

https://www.biorxiv.org/content/10.1101/2025.06.03.657760v1


---

## Raw CSV Format

Each row corresponds to a single particle detection.

### Required Columns

| Column                  | Description                  |
| ----------------------- | ---------------------------- |
| Frame                   | Frame number                 |
| X                       | X coordinate                 |
| Y                       | Y coordinate                 |
| Intensity               | Detection intensity          |
| Diffusion Coefficient   | Diffusion coefficient (DC)   |
| Receptor Density        | Receptor density (RD)        |
| Association Probability | Association probability (AP2)|
| Dissociation Rate       | Dissociation rate (DR2)      |
| Labeled Fraction        | Labeled fraction (LF)        |

### Additional Columns

#### Trimers

| Column |
| ------ |
| AP3    |
| DR3    |

#### Tetramers

| Column |
| ------ |
| AP3    |
| DR3    |
| AP4    |
| DR4    |

---

## Graph Generation

Graphs are generated using the graph-construction functions located in:

```text
src/deepfisik/data/
```

For each pair of consecutive frames:

1. Molecules are treated as graph nodes.
2. Pairwise distances are computed between detections in neighboring frames.
3. Edges are created between detections separated by less than a specified radius threshold.
4. Edge features are computed from:

   * spatial distance
   * intensity difference
5. The graph is trimmed to retain the most relevant connections.

---

## Graph Features

### Node Features

```text
[X, Y, Intensity]
```

### Edge Features

```text
[Distance, Intensity Difference]
```

### Graph Labels

#### Dimer Model

```text
DC
RD
AP2
DR2
LF
```

#### Trimer Model

```text
DC
RD
AP2
DR2
AP3
DR3
LF
```

#### Tetramer Model

```text
DC
RD
AP2
DR2
AP3
DR3
AP4
DR4
LF
```

---

## Output Format

Processed graphs are stored as PyTorch Geometric data objects:

```python
Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
)
```

and saved as:

```text
processed/
├── data_0.pt
├── data_1.pt
├── data_2.pt
└── ...
```

---

## Folder Structure

### Classification Datasets

```text
Datasets/
└── Images/
    ├── MonomerDataset/
    ├── DimerDataset/
    ├── TrimerDataset/
    └── TetramerDataset/
```

### Regression Datasets

```text
Datasets/
└── PureSimulations/
    ├── DimerDataset/
    ├── TrimerDataset/
    └── TetramerDataset/
```

---

## Loading a Dataset

```python
from deepfisik.data.datasetInteractionsReadAll import SMI

dataset = SMI(
    root="Datasets/PureSimulations/DimerDataset"
)
```

The dataset loader automatically reads all `.pt` files located in the `processed/` directory.

