# VSGL: Graph Property Detection Framework

This repository implements VSGL, a deep learning framework for graph property detection. The project includes dataset preprocessing, model definition and training, evaluation, and utility functions to streamline the workflow. It also provides baseline methods for comparison.

## Project Structure

```
.
├── main.py           # Main entry point for running experiments
├── train.py          # Training and evaluation logic
├── model.py          # Neural network architectures
├── utils.py          # Utility functions and helpers
├── dataset.py        # Dataset handling and preprocessing
├── dataset/          # Graph datasets 
└── baselines/        # Comparison baseline implementations
```

### Key Components

- **`main.py`**: Orchestrates the entire workflow, handling dataset loading, model initialization, training, and evaluation.
- **`train.py`**: Contains the core training and evaluation routines for both GAN and classifier models.
- **`model.py`**: Implements the neural network architectures, including:
  - Generator for graph layout generation
  - Discriminator for adversarial training
  - Multiple classifier options (ResNet, EfficientNet, ViT)
- **`utils.py`**: Provides utility functions for:
  - Graph visualization
  - Loss computation
  - Model state management
  - Performance metrics calculation
- **`dataset.py`**: Handles dataset operations:
  - Custom PyTorch Dataset classes
  - Data preprocessing
  - Graph property extraction
- **`baselines/`**: Contains implementations of comparison methods:
  - VN-Solver
  - Graphormer
  - Naive Bayes
  - Random baseline

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)

### Setup

1. Clone the repository:
```bash
git clone 
cd VSGL_framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare datasets:
```bash
# If dataset.zip is present, extract it
unzip dataset.zip
```

## Usage

### Basic Usage

To run the framework with default settings:

```bash
python main.py
```

### Customizing Parameters

You can modify various parameters in `main.py`:
- Sample size
- Number of training epochs
- Model architecture selection
- Training hyperparameters
- Evaluation metrics



## Features

### Core Capabilities
- **Flexible Dataset Handling**: Support for custom graph datasets with built-in preprocessing
- **Conditional GAN Framework**: Generate graph layouts based on specific properties
- **Multiple Classifier Options**: Choose from ResNet, EfficientNet, or Vision Transformer
- **Comprehensive Evaluation**: Compare against multiple baseline methods
- **Visualization Tools**: Built-in tools for graph and result visualization

### Baseline Methods
The framework includes several baseline implementations for comparison:
- **VN-Solver**: Traditional vision-based graph property solver
- **Graphormer**: State-of-the-art matrix-based graph model
- **Naive Bayes**: Simple probabilistic baseline
- **Random Baseline**: For performance benchmarking











