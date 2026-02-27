# CNN for Rock-Paper-Scissors Classification

A comparative study of three CNN architectures for hand gesture classification using PyTorch.


## Project Structure

```
├── data/
│   ├── raw/                    # Kaggle dataset (paper/rock/scissors)
│   └── custom_test/            # Real-world test images
├── notebooks/                  # Analysis notebooks (01-06)
├── src/                        # Source code (model.py, train.py, utils.py, evaluation.py)
├── models/                     # Trained models (.pth files)
├── reports/                    # Figures and results
└── report_pdf.tex                  # Detailed project report
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

Run notebooks in order:
1. `01_data_exploration.ipynb` - Dataset analysis
2. `02_baseline_simplecnn.ipynb` - Baseline model
3. `03_hyperparameter_tuning.ipynb` - Optimization
4. `04_advanced_architectures.ipynb` - Model comparison
5. `05_evaluation.ipynb` - Detailed metrics
6. `06_custom_testing.ipynb` -  Testing on real data

## Results

| Model | Test Acc |   Training Time |
|-------|----------|---------------|
| SimplestCNN with hyperparameters tuning| 96.35% |   433s |
| MediumCNN | 93.38% |   581s |
| DeepCNN | 95.66% |   1,038s |


## Technologies

PyTorch, NumPy, Matplotlib, scikit-learn, Jupyter