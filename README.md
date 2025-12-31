# Collaborative Process Parameter Recommender System

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2506.12252" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

This repository contains a Python implementation for "A collaborative process parameter recommender system for fleets of networked manufacturing machines --- with application to 3D printing". The codebase simulates collaborative and non-collaborative approaches to matrix completion using Alternating Least Squares (ALS) and evaluates their performance through various metrics including terminal cumulative regret and expected number of trials to optimal.

## Overview

The implementation consists of three main components organized in a modular pipeline:

1. **Data Generation** (`generate_random_matrix.py`): Generates synthetic matrices with controlled rank structure and masking patterns.
2. **Simulation** (`seq_fed_matrix_completion.py`): Executes sequential matrix completion experiments with both collaborative and non-collaborative strategies.
3. **Visualization** (`plot.py`): Creates LaTeX-formatted plots for comparative analyses.

## Project Structure

```
./
├── generate_random_matrix.py      # Data generation module
├── seq_fed_matrix_completion.py   # Main simulation engine
├── sequential_python_als.py         # ALS implementation
├── plot.py                        # Visualization module
│
├── data/                            # Generated datasets (auto-created)
│   ├── FPy_random_matrix_*.csv
│   ├── FPy_random_matrix_*_row_mapping.json
│   └── FPy_masking_metadata_*pct.json
│
├── results/                         # Simulation results (auto-created)
│   └── FRPy_result_tab_*.csv
│
└── log/                             # Selection logs (auto-created)
    ├── selection_log_collaborative_*.json
    └── selection_log_sequential_*.json
```

## Installation

### Prerequisites

- Python 3.7 or higher.
- pip package manager.

### Required Dependencies

Install all required Python packages using pip:

```bash
pip install numpy scipy pandas matplotlib scikit-learn lifelines tqdm
```

#### Detailed Package List

- **numpy** (≥1.19.0): Numerical computing and array operations.
- **scipy** (≥1.5.0): Scientific computing, including SVD and linear algebra.
- **pandas** (≥1.1.0): Data manipulation and analysis.
- **matplotlib** (≥3.3.0): Plotting and visualization.
- **scikit-learn** (≥0.23.0): Machine learning algorithms, including MICE imputation.
- **lifelines** (≥0.25.0): Survival analysis and Kaplan-Meier estimation.
- **tqdm** (≥4.50.0): Progress bar utilities.

For LaTeX rendering in plots (optional):

- macOS: ```brew install --cask mactex```.
- Ubuntu/Debian: ```sudo apt-get install texlive-full```.
- Windows: Install MiKTeX or TeX Live.


## Usage

### Pipeline Workflow

The complete experimental pipeline follows three steps:

#### Step 1: Data Generation

Generate synthetic matrices with controlled structure and masking patterns:

```bash
python generate_random_matrix.py
```

**Configuration Options:**

Edit the `GENERATION_MODE` variable in `generate_random_matrix.py`:

- `'random'`: Generate new synthetic matrices with group-based structure.
- `'load'`: Load existing matrices from `data` folder. E.g., load from `data/FPy_random_matrix_200_50_50.csv`.

**Output:**
- Test matrices with varying sizes (K) and ranks (r).
- Row mapping files linking test matrices to the full matrix.
- Masking metadata for multiple missing rates (40%, 50%, 60%, 70%, 80%).

#### Step 2: Run Simulations

Execute matrix completion experiments with configurable parameters:

```bash
python seq_fed_matrix_completion.py
```

**Key Parameters** (edit near the beginning of the file):

```python
num_machines = 50          # Number of machines (K)
intended_rank = 15         # Target rank (r)
num_pars = 50             # Number of configurations (l)
missing_rate = 80         # Sparsity percentage (ζ)
M_budget = 15             # Number of sequential trials (M)
lambda_val = 0.001        # ALS regularization parameter
small_matrix_rows = 5     # Number of parameter 1 (m_1)
```

**Output:**
- Result tables: `results/FRPy_result_tab_{K}_{l}_{r}_{r_est}_{ζ}_{M}.csv`;
- Selection logs (if enabled): `log/selection_log_{method}_{K}_{l}_{r}_{r_est}_{ζ}_{M}.json`.

#### Step 3: Generate Plots

Generate plots after completing all simulation cases:

```bash
python plot.py
```

**Configuration:**

Edit the `configs` list in the `__main__` block to specify which result files to plot. Each configuration compares different experimental conditions (rank, number of machines, or sparsity).

**Output:**
- Combined comparison plots: `Seq_MC_sim_combined_cases.png`

### Example Workflow

```bash
# 1. Generate data matrices
python generate_random_matrix.py

# 2. Run simulation case 1: K=50, l=50, r=5, ζ=60%, M=20
# Edit parameters in seq_fed_matrix_completion.py:
#   num_machines=50, intended_rank=5, missing_rate=60, M_budget=20
python seq_fed_matrix_completion.py

# 3. Run simulation case 2: K=50, l=50, r=10, ζ=60%, M=20
# Edit parameters in seq_fed_matrix_completion.py:
#   num_machines=50, intended_rank=10, missing_rate=60, M_budget=20
python seq_fed_matrix_completion.py

# ... (repeat for all simulation cases)

# 4. Generate plots after all simulations complete
python plot.py
```


## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wang2025collab,
      title={A Collaborative Process Parameter Recommender System for Fleets of Networked Manufacturing Machines -- with Application to 3D Printing}, 
      author={Weishi Wang and Sicong Guo and Chenhuan Jiang and Mohamed Elidrisi and Myungjin Lee and Harsha V. Madhyastha and Raed Al Kontar and Chinedum E. Okwudire},
      year={2025},
      eprint={2506.12252},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
}
```

## Contact

For questions or issues, please [open an issue](https://github.com/[username]/[repo]/issues) on GitHub.

