# MoPro: Model Training, Prediction, and Feature Visualization

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

ModProt is a flexible pipeline for training, prediction, and visualization of feature contributions in antimicrobial peptide activity modeling, including the estimation of Minimum Inhibitory Concentration (MIC). This framework accompanies and supports our work, “Gated Protein Language Modeling for Accurate Prediction of Antimicrobial Peptide Activity.”

For a comprehensive description of the approach, please refer to the paper: [link here]

---

## Features

- **Train** the MoPro model on one or more datasets.  
- **Predict** activity values and evaluate model performance on test datasets.  
- **Plot** feature contributions to interpret model predictions.  
- Flexible handling of multiple datasets via indices.  
- Modular design for easy extension to other models and datasets.  

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/PremSinghBist/abp.git
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt 
```


3. **Usage**


```bash
python train.py <mode> [--model MODEL_NAME] [--datasets DATASET_INDICES]
```
| Argument     | Description                                                            |
| ------------ | ---------------------------------------------------------------------- |
| `mode`       | Required. Mode to run: `train`, `predict`, or `plot`.                  |
| `--model`    | Optional. Model name. Default: `MoPro`.                                |
| `--datasets` | Optional. List of dataset indices (supports multiple). Default: `[0]`. |

4. **Examples:** 
```bash
Train mode (default dataset [0]):
python train.py train

Predict mode 
python train.py predict --datasets 0 1 2 3 4

Plot feature contributions for specific datasets:
python train.py plot --datasets 0
```
Plots are saved to ./visualization/IG/fold_<dataset_index>.


5. **Directory Structure:**
```bash
├── train.py                  # Main script with argparse
├── model/                    # Pretrained or saved models
├── data/                     # Complte datasets
├── visualization/
│   └── IG/                   # Feature contribution plots
├── requirements.txt          # Python dependencies
└── README.md
```
Notes:

The mode argument is mandatory. Use one of train, predict, or plot.
For visualization, the model must have been trained and saved beforehand.
Ensure all dependencies and datasets are properly prepared before running.


## Citation
If you use this code or pipeline in your research, please cite our paper:

Prem Singh Bist, et al. (2026). **. Journal/Conference Name. DOI: 10.xxxx/xxxx
