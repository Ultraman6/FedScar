## FedScar: Correcting Geometric Bias for Flatness-Consistent Federated Learning

This repository contains the implementation of the ICML 2026 submission **"FedScar: Correcting Geometric Bias for Flatness-Consistent Federated Learning"**.

The codebase provides:
- **Federated learning algorithms** implemented in `system/flcore/` (FedAvg, FedDyn, Scaffold, FedSAM, FedGamma, FedSMOO, FedLESAM, FedGMT, FedScar and its variants, etc.).
- **Client–server training framework** configurable from a single entry script `system/main.py`.
- **Image and text federated datasets and partition tools** under `dataset/` and `system/utils/data_utils.py`.
- **Flatness and loss landscape analysis utilities** (`system/utils/flat_metrics.py`, `system/utils/landscape.py`).

If you use this code, please cite our paper once it is publicly available.

---

### 1. Environment setup

- **Python**: recommended 3.9+  
- **CUDA**: a modern GPU and CUDA-capable PyTorch are recommended.

Install dependencies (no version pins; adjust as needed):

```bash
pip install -r requirements.txt
```

Main libraries used include:
- `torch`, `torchvision`, `torchtext`
- `numpy`, `scipy`, `pandas`, `scikit-learn`
- `tqdm`, `matplotlib`, `seaborn`
- `pillow`, `h5py`
- `ujson`, `openpyxl`

---

### 2. Project structure

- `dataset/`
  - `main.py`: scripts to generate and partition datasets (e.g., CIFAR-10/100, CINIC-10, AGNews) for federated learning.
  - `utils/`: dataset utilities, long-tailed CIFAR variants, Dirichlet partitioning, etc.
- `system/`
  - `main.py`: main entry point to run federated learning experiments.
  - `flcore/`: server and client implementations for different FL algorithms.
  - `utils/`: memory usage utilities, data loading, flatness metrics, loss landscape visualization, optimizer helpers, etc.

Data partitions are typically stored in a separate directory structure (e.g., `.../dataset/data/<dataset_name>/seed.../`), with a `config.json` describing partition metadata.

---

### 3. Preparing datasets and data partitions

Dataset preparation is handled by `dataset/main.py` and utilities under `dataset/utils/`:
- Standard datasets: CIFAR-10/100, CINIC-10, FMNIST (vision) and AGNews (NLP).
- Non-IID / long-tailed settings: generated via Dirichlet sampling or shard-based partitioning.

Typical steps:
1. Use the scripts in `dataset/` to download raw datasets and generate federated data partitions under a root such as `dataset/data/`.
2. Each data partition directory contains:
   - `config.json`: metadata (number of clients, number of classes, seed, etc.).
   - `train/` and `test/` subdirectories with per-client data files.
3. In `system/main.py`, the argument `--data_partition` (a relative path like `cifar10/seed1_nc100_LDA_alpha0.01_imb0.5`) is combined with a root path to locate the partition directory.

You may need to adjust the base path used in `system/main.py` (the `args.data_partition_dir` root) to match your local directory layout.

---

### 4. Running experiments

All FedScar and baseline experiments are launched via:

```bash
cd system
python main.py [ARGS...]
```

Key arguments in `system/main.py`:
- **Federated setting**
  - `-gr`, `--global_rounds`: number of global communication rounds (default: 1000).
  - `-le`, `--local_epochs`: local epochs per client update (can be a list).
  - `-lbs`, `--batch_size`: local batch size (can be a list).
  - `-lr`, `--local_learning_rate`: local learning rate.
  - `-jr`, `--join_ratio`: client participation ratio (fraction of clients per round).
  - `-dev`, `--device`: device string (e.g., `cuda:0` or `cpu`).
  - `--momentum`, `--weight_decay`, `--lr_decay`: optimizer hyperparameters.
- **Data & partition**
  - `-data`, `--dataset`: dataset name (`cifar10`, `cifar100`, `cinic10`, `fminist`, `agnews`, etc.).
  - `--data_partition`: list of data partition directory names (relative to the chosen root).
- **Analysis**
  - `-rl`, `--round_list`: list of global rounds at which to compute sharpness or visualize landscapes.
  - `-cs`, `--cal_sharp`: whether to compute sharpness metrics.
  - `-ll`, `--loss_land`: whether to perform loss landscape visualization.
  - `-sm`, `--save_model`, `-sl`, `--save_local`: model saving options.

The concrete algorithm(s) to run and their hyperparameters are defined in the list `ALGORITHM_CONFIGS` in `system/main.py`. Each entry is of the form:

```python
(FedScar, {'alpha': 0.1, 'gama': 0.9, 'beta': 0.06, 'rho': [0.1]}, '2511220058')
```

where the last string is an experiment timestamp or tag used for logging and saving.

---

### 5. Example commands

#### 5.1 CIFAR-10 FedAvg baseline

1. Configure `ALGORITHM_CONFIGS` to include FedAvg, e.g.:

```python
ALGORITHM_CONFIGS = [
    (FedAvg, {}, 'baseline_fedavg'),
]
```

2. Run:

```bash
cd system
python main.py \
  --dataset cifar10 \
  --data_partition cifar10/seed1_nc100_LDA_alpha0.01_imb0.5 \
  --global_rounds 1000 \
  --local_epochs 5 \
  --batch_size 50 \
  --local_learning_rate 0.1 \
  --join_ratio 0.1 \
  --device cuda:0
```

#### 5.2 CIFAR-10 FedScar (geometric-bias-corrected, flatness-consistent)

1. In `ALGORITHM_CONFIGS`, enable FedScar with appropriate hyperparameters:

```python
ALGORITHM_CONFIGS = [
    (FedScar, {'alpha': 0.1, 'gama': 0.9, 'beta': 0.06, 'rho': [0.1]}, 'fedscar_main'),
]
```

2. Run:

```bash
cd system
python main.py \
  --dataset cifar10 \
  --data_partition cifar10/seed1_nc100_LDA_alpha0.01_imb0.5 \
  --global_rounds 1000 \
  --local_epochs 5 \
  --batch_size 50 \
  --local_learning_rate 0.1 \
  --join_ratio 0.1 \
  --device cuda:0
```

#### 5.3 Sharpness and loss landscape analysis

To compute sharpness or visualize loss landscapes at specific rounds:

```bash
cd system
python main.py \
  --dataset cifar10 \
  --data_partition cifar10/seed1_nc100_LDA_alpha0.01_imb0.5 \
  --round_list 200 500 800 \
  --cal_sharp True \
  --loss_land False
```

or set `--loss_land True` to enable visualization (see `Server` methods `sharpness_v2` and `visualize_loss_landscape`).

---

### 6. Logging, checkpoints, and flatness metrics

Each algorithm/server class in `system/flcore/servers/` handles:
- Logging of training and testing metrics over global rounds.
- Saving models and relevant statistics at configured intervals (`--save_gap`, `--eval_round`).

Flatness-related metrics and loss landscapes are computed using:
- `system/utils/flat_metrics.py` (Hessian-based metrics, gradient norms, low-pass filters).
- `system/utils/landscape.py` (2D/3D loss landscape visualization).

Outputs are typically written to experiment-specific directories determined by the algorithm name, dataset, data partition, and timestamp string in `ALGORITHM_CONFIGS`.

---

### 7. License and contact

This code is released for **academic research** on federated learning, flatness-aware optimization, and geometric bias correction.  
For questions about the code or paper, please contact the authors of **"FedScar: Correcting Geometric Bias for Flatness-Consistent Federated Learning"** (see the ICML 2026 submission for details).

