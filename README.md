# MoPro: Model Training, Prediction, and Feature Visualization

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MoPro is a flexible pipeline for training, predicting, and visualizing feature contributions for peptide/protein modeling tasks. It supports multiple datasets, cross-validation, and interpretable results using feature contribution plots (Integrated Gradients).

---

## Features

- **Train** the MoPro model on one or more datasets.  
- **Predict** activity values and evaluate model performance on test datasets.  
- **Plot** feature contributions to interpret model predictions.  
- Flexible handling of multiple datasets via indices.  
- Modular design for easy extension to other models and datasets.  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mopro.git
cd abp

2. Install dependencies:
pip install -r requirements.txt


Usage
python train.py <mode> [--model MODEL_NAME] [--datasets DATASET_INDICES]

| Argument     | Description                                                            |
| ------------ | ---------------------------------------------------------------------- |
| `mode`       | Required. Mode to run: `train`, `predict`, or `plot`.                  |
| `--model`    | Optional. Model name. Default: `MoPro`.                                |
| `--datasets` | Optional. List of dataset indices (supports multiple). Default: `[0]`. |

Examples 
Train mode (default dataset [0]):
python train.py train

Predict mode 
python train.py predict --datasets 0 1 2 3 4

Plot feature contributions for specific datasets:
python train.py plot --datasets 0

Plots are saved to ./visualization/IG/fold_<dataset_index>.

Directory Structure

.
├── train.py                       # Main script with argparse
├── model/                       # Pretrained or saved models
├── data/                     # Complte datasets
├── visualization/
│   └── IG/                       # Feature contribution plots
├── requirements.txt               # Python dependencies
└── README.md

Notes

The mode argument is mandatory. Use one of train, predict, or plot.

For visualization, the model must have been trained and saved beforehand.

Ensure all dependencies and datasets are properly prepared before running.


## Citation
If you use this code or pipeline in your research, please cite our paper:

Prem Singh Bist, et al. (2026). **. Journal/Conference Name. DOI: 10.xxxx/xxxx



