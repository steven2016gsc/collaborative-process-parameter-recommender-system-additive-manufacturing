# Collaborative Process Parameter Recommender System

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2506.12252" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

This repository contains a Python implementation for "[A collaborative process parameter recommender system for fleets of networked manufacturing machines --- with application to 3D printing](https://arxiv.org/abs/2506.12252)". The codebase simulates collaborative and non-collaborative approaches to matrix completion using Alternating Least Squares (ALS) and evaluates their performance through various metrics including terminal cumulative regret and expected number of trials to optimal.

## Overview

The implementation consists of three main components organized in a modular pipeline:

1. **Data Generation** (`generate_random_matrix.py`): Generates synthetic matrices with controlled rank structure and masking patterns for simulation studies.
2. **Simulation** (`seq_fed_matrix_completion_sim.py`): Runs sequential matrix completion for simulation studies (Section 3) with both collaborative and non-collaborative strategies.
2. **Experiments** (`seq_fed_matrix_completion_exp.py`): Runs sequential matrix completion for experimental data (Section 4) with both collaborative and non-collaborative strategies.
3. **Plotting** (`plot.py`): Creates LaTeX-formatted plots for comparative analyses.

## Project Structure

```
./
├── generate_random_matrix.py      # Data generation (simulation)
├── seq_fed_matrix_completion_sim.py   # Main - simulation studies
├── seq_fed_matrix_completion_exp.py   # Main - experimental studies
├── sequential_python_als.py         # ALS implementation
├── plot.py                        # Plotting
│
├── data/                            # Datasets
│   ├── FPy_random_matrix_*.csv
│   ├── FPy_random_matrix_*_row_mapping.json
│   ├── FPy_masking_metadata_*pct.json
│   ├── full_*_mat.txt
│   ├── print_time_matr.txt
│   └── nU_exp.csv
│
├── results/                         # Results
│   ├── FRPy_result_tab_*.csv                           # Simulation result
│   ├── cumu_regrets_largest_prediction.csv             # Cumulative regrets (collaborative)
│   ├── cumu_regrets_solo_largest_prediction.csv        # Cumulative regrets (non-collaborative)
│   ├── sub_cumu_chosen.csv                             # Log of Case 2 machine selections (collaborative)
│   ├── sub_cumu_chosen_no_colab.csv                    # Log of Case 2 machine selections (non-collaborative)
│   ├── sub_cumu_regrets_largest_prediction.csv         # Case 2 cumulative regrets (collaborative)
│   └── sub_cumu_regrets_solo_largest_prediction.csv    # Case 2 cumulative regrets (non-collaborative)
│
├── figures/                         # Generated plots
│   ├── Seq_MC_sim_combined_cases.png                   # Combined simulation cases comparison
│   ├── Seq_matrix_completion_case_1.png                # Case 1 plot
│   └── Seq_matrix_completion_case_2.png                # Case 2 plot
│
└── log/                             # Selection logs (simulated studies)
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

For experimental studies (`seq_fed_matrix_completion_exp.py`), additional packages are required:

```bash
pip install pyspark recommenders
```

#### Detailed Package List

**Core Dependencies (Required for all workflows):**

- **numpy** (≥1.19.0): Numerical computing, array operations, linear algebra (QR decomposition, SVD), and matrix manipulations.
- **scipy** (≥1.5.0): Scientific computing, including SVD via `scipy.linalg.svd` for rank estimation.
- **pandas** (≥1.1.0): Data manipulation, CSV I/O, and tabular data structures.
- **matplotlib** (≥3.3.0): Plotting and visualization with LaTeX rendering support for publication-quality figures.
- **scikit-learn** (≥0.23.0): Machine learning utilities including:
  - MICE (Multivariate Imputation by Chained Equations) via `IterativeImputer`
  - Pairwise distance metrics (`pairwise_distances`, `rbf_kernel`)
  - Used in rank estimation algorithm
- **lifelines** (≥0.25.0): Survival analysis and Kaplan-Meier estimation for computing expected number of trials to optimal.
- **tqdm** (≥4.50.0): Progress bar utilities for tracking long-running experiments.

**Additional Dependencies (Experimental studies only):**

- **pyspark** (≥3.0.0): Apache Spark distributed computing framework. Required for:
  - ALS (Alternating Least Squares) collaborative filtering via `pyspark.ml.recommendation.ALS`
  - Distributed dataframe operations via `pyspark.sql`
  - Large-scale matrix factorization in experimental workflow
- **recommenders** (≥1.1.0): Microsoft Recommenders library providing utilities for recommendation systems:
  - Spark session management (`start_or_get_spark`)
  - Evaluation metrics (`SparkRatingEvaluation`, `SparkRankingEvaluation`)
  - Dataset utilities and splitting functions

### Additional System Requirements

**Java (Required for PySpark):**

PySpark 3.0+ requires Java 8 or later to be installed. 

- **macOS**:
  ```bash
  brew install openjdk@17
  ```
  Then add to your shell profile (e.g., `~/.zshrc` or `~/.bash_profile`):
  ```bash
  export JAVA_HOME=$(/usr/libexec/java_home -v 17)
  export PATH=$JAVA_HOME/bin:$PATH
  ```

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install openjdk-17-jdk
  ```

- **Windows**: Download and install Java 17 from [Oracle JDK](https://www.oracle.com/java/technologies/downloads/) or [Adoptium OpenJDK](https://adoptium.net/)

Verify Java installation:
```bash
java -version
```

**Note**: Java 11 also works if you already have it installed. For PySpark 3.0-3.2, Java 8 (≥8u201) is also supported.

**LaTeX (Optional for plot rendering):**

- macOS: ```brew install --cask mactex```.
- Ubuntu/Debian: ```sudo apt-get install texlive-full```.
- Windows: Install MiKTeX or TeX Live.


## Usage

### Pipeline Workflow (simulation studies)

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
python seq_fed_matrix_completion_sim.py
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

### Example Workflow (Simulation Studies)

```bash
# 1. Generate data matrices
python generate_random_matrix.py

# 2. Run simulation case 1: K=50, l=50, r=5, ζ=60%, M=20
# Edit parameters in seq_fed_matrix_completion.py:
#   num_machines=50, intended_rank=5, missing_rate=60, M_budget=20
python seq_fed_matrix_completion_sim.py

# 3. Run simulation case 2: K=50, l=50, r=10, ζ=60%, M=20
# Edit parameters in seq_fed_matrix_completion.py:
#   num_machines=50, intended_rank=10, missing_rate=60, M_budget=20
python seq_fed_matrix_completion_sim.py

# ... (repeat for all simulation cases)

# 4. Generate plots after all simulations complete
python plot.py
```

### Running Experimental Studies

For experimental studies using experimental data (Section 4), ensure you have the experimental data file `nU_exp.csv` in the `data/` directory, then run:

```bash
python seq_fed_matrix_completion_exp.py
```

**Note**: This requires PySpark and Java to be installed (see [Additional System Requirements](#additional-system-requirements)).


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

