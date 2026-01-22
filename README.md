# VSAL: Vision-based Graph Property Detection Framework
[![DOI](https://zenodo.org/badge/920212424.svg)](https://doi.org/10.5281/zenodo.18332924)

This repository implements VSAL (Vision neural Solver with Adaptive Layout), a deep learning framework for graph property detection. The project includes dataset preprocessing, model definition and training, evaluation, and utility functions to streamline the workflow. It also provides comprehensive baseline methods for comparison.

## Project Structure

```
.
├── DenseGCN_based_VSAL/          # DenseGCN-based implementation
│   ├── main.py                   # Main entry point for DenseGCN experiments
│   ├── train.py                  # Training and evaluation logic
│   ├── model.py                  # Neural network architectures
│   ├── utils.py                  # Utility functions and helpers
│   └── dataset.py                # Dataset handling and preprocessing
├── Graphormer_based_VSAL/        # Graphormer-based implementation
│   ├── main.py                   # Main entry point for Graphormer experiments
│   ├── train.py                  # Training and evaluation logic
│   ├── model.py                  # Neural network architectures
│   ├── utils.py                  # Utility functions and helpers
│   └── dataset.py                # Dataset handling and preprocessing
├── Dataset/                      # Graph datasets for different tasks
│   ├── claw-free_graph_classification/
│   ├── hamiltonian_cycle_problem/
│   ├── planarity_verification/
│   └── tree_recognition/
├── baselines/                    # Comparison baseline implementations
│   ├── Graphormer/
│   ├── Graphormer-GD/
│   ├── EquiformerV2.md
│   ├── GraphsGPT.md
│   ├── Naive_bayes/
│   ├── Random_guess/
│   └── VN-solver/
└── generate_new_graphs/          # Scripts for producing large and huge graphs
    ├── generate_new_claw_free_graphs.py
    ├── generate_new_hamitonian_graphs.py
    ├── generate_new_planar_graphs.py
    └── generate_new_tree_graphs.py
```

### Key Components

#### Implementation Modules
- **`DenseGCN_based_VSAL/`**: Complete implementation using DenseGCN architecture
- **`Graphormer_based_VSAL/`**: Complete implementation using Graphormer architecture

Each implementation module contains:
- **`main.py`**: Orchestrates the entire workflow, handling dataset loading, model initialization, training, and evaluation
- **`train.py`**: Contains the core training and evaluation routines for both GAN and classifier models
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

#### Supporting Components
- **`Dataset/`**: Contains datasets for four graph property detection tasks:
  - Claw-free graph classification
  - Hamiltonian cycle problem
  - Planarity verification
  - Tree recognition
- **`baselines/`**: Contains implementations of comparison methods:
  - VN-Solver
  - Graphormer
  - Graphormer-GD
  - EquiformerV2
  - GraphsGPT
  - Naive Bayes
  - Random baseline
- **`generate_new_graphs/`**: Contains multiple scripts to produce large and huge graphs for different tasks:
  - Generate new Hamiltonian and non-Hamiltonian graphs
  - Generate new planar and non-planar graphs
  - Generate new claw-free and non-claw-free graphs
  - Generate new tree and non-tree graphs

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)
- PyTorch with CUDA support

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd framework-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare datasets:
```bash
# Extract datasets from the Dataset/ directory
cd Dataset/
# Unzip the specific dataset you want to use
# For example, for Hamiltonian cycle problem:
unzip hamiltonian_cycle_problem/small_hamiltonian_graphs_dataset.zip
```

## Usage

### Basic Usage

The framework provides two main implementations. Choose the one that best fits your needs:

#### DenseGCN-based Implementation

To run the DenseGCN-based VSAL framework:

```bash
cd DenseGCN_based_VSAL
python main.py
```

#### Graphormer-based Implementation

To run the Graphormer-based VSAL framework:

```bash
cd Graphormer_based_VSAL
python main.py
```

### Customizing Parameters

You can modify various parameters in the respective `main.py` files:
- Sample size
- Number of training epochs
- Model architecture selection
- Training hyperparameters
- Evaluation metrics
- Dataset selection (Hamiltonian, Planar, Claw-free, Tree)

### Graph Property Detection Tasks

The framework supports four main graph property detection tasks:

1. **Hamiltonian Cycle Problem**: Detect if a graph contains a Hamiltonian cycle
2. **Planarity Verification**: Determine if a graph is planar
3. **Claw-free Graph Classification**: Classify graphs as claw-free or not
4. **Tree Recognition**: Identify if a graph is a tree structure

### Dataset Sizes

Each task includes datasets of different sizes:
- **Small**: 4-20 nodes graphs (from House of Graph)
- **Medium**: 21-50 nodes graphs (from House of Graph)
- **Large**: 401-500 nodes graphs (Produced by us)
- **Huge**: 901-1000 nodes graphs (Produced by us)



### Baseline Methods
The framework includes comprehensive baseline implementations for comparison:
- **VN-Solver**
- **Graphormer**
- **Graphormer-GD**
- **EquiformerV2**
- **GraphsGPT**
- **Naive Bayes**
- **Random Baseline**






