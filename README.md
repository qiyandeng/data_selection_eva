# Data Selection Testbed

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The first open-source testbed for data selection that implements 15 representative strategies, spanning model-agnostic/model-aware and task-agnostic/task-aware strategies for efficient and effective data selection. Our testbed conducts evaluation on 12 datasets with various data distributions and in rich scenarios with various user-specified parameters.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Deep Learning Data Selection](#deep-learning-data-selection)
- [LLMs SFT Data Selection](#llms-sft-data-selection)
- [Datasets](#datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This testbed provides a comprehensive framework for evaluating data selection strategies across diverse datasets and model architectures. It consists of three main modules:

1. **Configuration Loader**: Allows users to configure datasets, models, parameters, data selection strategies, and experimental controls (e.g., logging).

2. **Data Selector**: Offers 15 data selection strategies for efficient and effective training. It runs a data selection strategy specified in the user configuration to select a subset of data and passes it to the Model Evaluator. For iterative strategies, the Data Selector and Model Evaluator interact to continuously improve the model performance.

3. **Model Evaluator**: Trains the target model using the selected data subset and evaluates its performance on the test set, measuring both accuracy and latency.

![Workflow](./image/workflow-ds_00.png)

## 🏗️ Architecture

The testbed is designed with a modular architecture that enables seamless switching between different methods and makes it easy to integrate new strategies for benchmarking or research purposes. The modular design facilitates easy integration of new data selection methods, datasets, and model architectures.

## 🚀 Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd data_selection_lib

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

```python
import torch
from torch.utils.data import DataLoader
import nets
from mydatasets import adult
from methods.kmeans import KMeansSelection
from model_utils import train_model, evaluate_model

# Basic configuration
data_path = "mydatasets"
method = 'kmeans'
task_type = 'classification'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test = adult(data_path, balance_target=True)

# Initialize model
base_model = nets.__dict__['MLP'](
    channel=channel,
    num_classes=num_classes,
    im_size=im_size,
    pretrained=False
)

# Data selection
selector = KMeansSelection()
core_loader = selector.select(
    model=base_model,
    selection_fraction=0.1,
    k=num_classes,
    train_set=dst_train,
    task_type=task_type,
    device=device
)

# Train and evaluate
trained_model = train_model(
    model=base_model,
    train_loader=core_loader,
    val_loader=DataLoader(dst_val, batch_size=128, shuffle=False),
    device=device,
    task_type=task_type,
    epochs=100,
    patience=10
)

evaluate_model(
    model=trained_model,
    test_loader=DataLoader(dst_test, batch_size=128, shuffle=False),
    device=device,
    task_type=task_type
)
```

## 🧠 Deep Learning Data Selection

Our testbed implements 15 representative data selection strategies covering both efficient and effective training approaches:

### Implemented Methods

| Method | Category | Description | Implementation |
|--------|----------|-------------|----------------|
| **K-Means** | Model-agnostic | Clustering-based selection using K-means algorithm | ✅ [selection/kmeans.py](selection/kmeans.py) |
| **Herding** | Model-agnostic | Core-set selection using herding algorithm | ✅ [methods/herding.py](methods/herding.py) |
| **K-Center Greedy** | Model-agnostic | Greedy k-center clustering for core-set selection | ✅ [methods/kcentergreedy.py](methods/kcentergreedy.py) |
| **Low Dimension** | Model-agnostic | Selection in low-dimensional space | ✅ [methods/lowdimention.py](methods/lowdimention.py) |
| **Random** | Model-agnostic | Random sampling baseline | ✅ [methods/random.py](methods/random.py) |
| **Full** | Model-agnostic | Full dataset training baseline | ✅ [methods/full.py](methods/full.py) |
| **CRAIG** | Model-aware | Gradient-based core-set selection | ✅ [methods/craig.py](methods/craig.py) |
| **GradMatch** | Model-aware | Gradient matching for data selection | ✅ [methods/gradmatch.py](methods/gradmatch.py) |
| **GLISTER** | Model-aware | Generalization based data subset selection | ✅ [methods/glister.py](methods/glister.py) |
| **CREST** | Model-aware | Core-set selection with gradient-based scoring | ✅ [methods/crest.py](methods/crest.py) |
| **TracIn** | Model-aware | Training data influence estimation | ✅ [methods/TracIn.py](methods/TracIn.py) |
| **LiSSA** | Model-aware | Linear time second-order algorithm | ✅ [methods/LiSSA.py](methods/LiSSA.py) |
| **Arnoldi** | Model-aware | Arnoldi iteration for influence computation | ✅ [methods/Arnoldi.py](methods/Arnoldi.py) |
| **TMCS** | Model-aware | Training data influence with Monte Carlo sampling | ✅ [methods/TMCS.py](methods/TMCS.py) |
| **KNN-Shapley** | Model-aware | K-nearest neighbors Shapley value estimation | ✅ [methods/KNN_shapley.py](methods/KNN_shapley.py) |
| **CS-Shapley** | Model-aware | Core-set Shapley value computation | ✅ [methods/cs_shapley.py](methods/cs_shapley.py) |
| **G-Shapley** | Model-aware | Gradient-based Shapley value estimation | ✅ [methods/G_shapley.py](methods/G_shapley.py) |
| **DVRL** | Model-aware | Data valuation using reinforcement learning | ✅ [methods/dvrl.py](methods/dvrl.py) |
| **DBAL** | Model-aware | Deep batch active learning | ✅ [methods/dbal.py](methods/dbal.py) |
| **DFAL** | Model-aware | Deep fool active learning | ✅ [methods/dfal.py](methods/dfal.py) |
| **Boundary Aware** | Model-aware | Boundary-aware data selection | ✅ [methods/boundary_aware.py](methods/boundary_aware.py) |
| **CG-Influence** | Model-aware | Conjugate gradient influence estimation | ✅ [methods/cg_influence.py](methods/cg_influence.py) |
| **SVP Max Entropy** | Model-aware | Support vector perturbation with maximum entropy | ✅ [methods/svp_max_entropy.py](methods/svp_max_entropy.py) |
| **IAS** | Model-aware | Influence-aware subset selection | ✅ [methods/IAS.py](methods/IAS.py) |

### Supported Models

The testbed supports various model architectures:

- **CNNs**: ResNet, VGG, AlexNet, InceptionV3, MobileNetV3, WideResNet, ViT
- **MLPs**: Standard MLP, TabNet, TabTransformer
- **RNNs**: LSTM
- **Transformers**: BERT, T5
- **Specialized**: DSN, EfficientNet

### Supported Datasets

#### In-Distribution (ID) Datasets
1. **CIFAR-100** - Large-scale image classification
2. **TinyImageNet** - Image classification with rich visual diversity
3. **Adult** - Census income prediction (tabular)
4. **IMDB-Large** - Large-scale text classification (>20M records)
5. **Covtype** - Large-scale multiclass classification (tabular)
6. **Bike** - Regression benchmark (tabular)
7. **IMDB** - Binary text classification
8. **News** - Multiclass text classification

#### Out-of-Distribution (OOD) Datasets
1. **MNIST→MNIST-M** - Domain adaptation (synthetic backgrounds)
2. **MNIST-M→MNIST** - Reverse domain adaptation
3. **HR** - Employee attrition prediction (cross-departmental)
4. **House** - House price prediction (cross-city)

## 🤖 LLMs SFT Data Selection

*This section is reserved for Large Language Models Supervised Fine-Tuning data selection methods.*

### Methods Comparison Table

| Method | Open Source | Repository/Implementation | Notes |
|--------|-------------|---------------------------|-------|
| **Method 1** | ✅ | [Link to implementation] | Official implementation |
| **Method 2** | ❌ | [Our implementation] | Self-implemented based on paper |
| **Method 3** | ✅ | [Link to implementation] | Official implementation |
| **Method 4** | ❌ | [Our implementation] | Self-implemented based on paper |

*Note: This table will be populated with LLM-specific data selection methods for supervised fine-tuning scenarios.*

## 📊 Datasets

### Dataset Statistics

| Dataset | Type | Modality | Classes | Train Size | Test Size | Distribution |
|---------|------|----------|---------|------------|-----------|--------------|
| CIFAR-100 | Classification | Image | 100 | 50,000 | 10,000 | ID |
| TinyImageNet | Classification | Image | 200 | 100,000 | 10,000 | ID |
| Adult | Classification | Tabular | 2 | 32,561 | 16,281 | ID |
| IMDB-Large | Classification | Text | 2 | 20,000,000+ | 2,500,000 | ID |
| Covtype | Classification | Tabular | 7 | 464,809 | 116,203 | ID |
| Bike | Regression | Tabular | - | 13,886 | 3,472 | ID |
| IMDB | Classification | Text | 2 | 25,000 | 25,000 | ID |
| News | Classification | Text | 4 | 113,736 | 28,434 | ID |
| MNIST→MNIST-M | Classification | Image | 10 | 60,000 | 10,000 | OOD |
| MNIST-M→MNIST | Classification | Image | 10 | 60,000 | 10,000 | OOD |
| HR | Classification | Tabular | 2 | 10,000 | 2,500 | OOD |
| House | Regression | Tabular | - | 15,000 | 3,750 | OOD |

### Data Distribution

- **In-Distribution (ID)**: Training and test sets share the same data distribution
- **Out-of-Distribution (OOD)**: Training and test sets come from different distributions
- **Train/Val/Test Split**: 80%/10%/10% for ID datasets
- **OOD Split**: Following [Data Shapley](https://arxiv.org/abs/1904.02868) methodology

## 📈 Evaluation Metrics

### Effectiveness Metrics

- **Classification Tasks**: F1-score
  - F1-score = 2 × (Precision × Recall) / (Precision + Recall)
- **Regression Tasks**: Mean Squared Error (MSE)
  - MSE = (1/n) × Σ(y - y_predict)²

### Efficiency Metrics

- **Data Selection Time**: Time required for subset selection
- **Model Training Time**: Total training time with selected subset


## 🔧 Configuration

The testbed uses YAML configuration files for easy customization:

```yaml
# Example configuration
dataset:
  name: "adult"
  path: "mydatasets"
  balance_target: true

model:
  name: "MLP"
  pretrained: false

selection:
  method: "kmeans"
  fraction: 0.1
  parameters:
    k: 2

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
  patience: 10

experiment:
  device: "cuda"
  seed: 42
  log_level: "INFO"
```

## 📚 Usage Examples

### Running a Single Experiment

```bash
python main.py --method kmeans --dataset adult --model MLP --selection_fraction 0.1
```

### Running Multiple Methods

```bash
# Run all methods on a dataset
python run_experiments.py --dataset adult --methods kmeans,herding,craig,glister
```

### Custom Dataset Integration

```python
from mydatasets import YourDataset
from methods.your_method import YourSelectionMethod

# Load your dataset
dataset = YourDataset(data_path)

# Use your selection method
selector = YourSelectionMethod()
selected_data = selector.select(dataset, fraction=0.1)
```

## 🏃‍♂️ Running Experiments

### Hardware Requirements

- **GPU**: 4× NVIDIA RTX 3090 (24GB each)
- **CPU**: Dual Intel Xeon Gold 6148 (80 threads total)
- **RAM**: 1 TiB system memory

### Hyperparameter Optimization

We employ [Optuna](https://optuna.org/) for hyperparameter optimization using grid search. See Appendix for detailed hyperparameter configurations.

<!-- ## 📖 Citation

If you find this testbed useful in your research, please cite our paper:

```bibtex
@article{your-paper-2024,
  title={Your Paper Title},
  author={Your Name and Co-authors},
  journal={Conference/Journal Name},
  year={2024}
}
``` -->

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Adding New Methods

1. Create a new file in the `methods/` directory
2. Implement the selection interface
3. Add the method to `import_methods.py`
4. Update the documentation

### Adding New Datasets

1. Create a new file in the `mydatasets/` directory
2. Implement the dataset interface
3. Add preprocessing and loading functions
4. Update the dataset registry

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the authors of the original papers for their valuable contributions to the field of data selection. This testbed builds upon their foundational work.

## 📞 Contact

For questions and issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This is the first open-source testbed for comprehensive data selection evaluation. We hope it will facilitate research and development in the field of efficient and effective data selection for machine learning. 