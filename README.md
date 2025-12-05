# OpenSLR - Open Sign Language Recognition Framework

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

OpenSLR (Open Sign Language Recognition) is a comprehensive, modular framework for continuous sign language recognition built on PyTorch. It provides a flexible and extensible architecture that supports multiple state-of-the-art model architectures and training strategies, making it suitable for both research and production use cases.

## Key Features

- **Modular Design**: Highly decoupled components with a manager-based architecture for easy extension and maintenance
- **Multiple Model Architectures**: Support for SlowFast, TLP (Two-Stream Lightweight Pyramid), VAC (Visual Attention Consistency), CorrNet, and other advanced models
- **Efficient Training**: Mixed precision training, memory-mapped data loading, and multi-GPU support
- **Professional Evaluation**: Word Error Rate (WER) metrics, Beam Search decoding, and detailed result analysis
- **Experiment Management**: Weights & Biases integration for experiment tracking and visualization
- **Multiple Dataset Support**: Phoenix2014, Phoenix2014-T, CSL, CSL-Daily, and customizable dataset support

## Project Structure

```
OpenSLR/
├── core/                  # Main source code
│   ├── main.py           # Program entry point
│   ├── manager/          # Manager components
│   │   ├── argument_manager.py
│   │   ├── config_manager.py
│   │   ├── experiment_manager.py
│   │   └── ...
│   ├── models/           # Model architectures
│   │   ├── build_function.py
│   │   ├── modules/
│   │   └── senmodules/
│   ├── dataset/          # Dataset loaders
│   ├── libs/             # External libraries and utilities
│   ├── configs/          # Configuration files
│   └── preprocess/       # Data preprocessing scripts
└── docs/                 # Documentation
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/OpenSLR.git
cd OpenSLR

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

OpenSLR supports several public sign language datasets. For example, to use the Phoenix2014 dataset:

```bash
# Download and preprocess the Phoenix2014 dataset
cd core/preprocess
python dataset_preprocess.py --dataset phoenix2014 --dataset-root /path/to/phoenix2014
```

### 2. Configure Training

Create or modify a configuration file in `core/configs/`:

```yaml
feeder: dataset.dataloader_video.VideoDataset
phase: train
dataset: phoenix2014
num_epoch: 80
work_dir: ./work_dir/baseline_experiment/
batch_size: 8

device: 0,1  # GPU devices
eval_interval: 1
save_interval: 5

model: models.build_function.build_slowfast
model_args:
    num_classes: 1296
    hidden_size: 1024
    c2d_type: slowfast101
    kernel_size: ['K5', "P2", 'K5', "P2"]

optimizer_args:
    optimizer: Adam
    base_lr: 0.0001
    step: [40, 60]
    weight_decay: 0.0001
```

### 3. Start Training

```bash
cd core
python main.py --config configs/baseline.yaml --work-dir ./work_dir/baseline_experiment
```

### 4. Model Inference

```python
from OpenSLR import infer

# Load video data (shape: [batch, channels, frames, height, width])
video_data = ...  # Your video data here
video_length = ...  # Length of the video sequence

# Perform sign language recognition
recognized_sents = infer(video_data, video_length)
print("Recognition Result:", recognized_sents)
```

## Model Architectures

OpenSLR supports various state-of-the-art model architectures:

- **SlowFast**: Two-pathway network for video recognition with different temporal resolutions
- **TLP (Two-Stream Lightweight Pyramid)**: Efficient architecture with spatial and temporal streams
- **VAC (Visual Attention Consistency)**: Incorporates attention mechanisms for improved performance
- **CorrNet**: Correlation-based network for sign language recognition
- **Custom Models**: Easily extendable to add new model architectures

## Datasets

OpenSLR supports several public sign language datasets:

- **Phoenix2014**: Large-scale continuous sign language recognition dataset
- **Phoenix2014-T**: German sign language dataset with temporal annotations
- **CSL**: Chinese Sign Language dataset
- **CSL-Daily**: Daily Chinese Sign Language dataset

## Evaluation

The framework provides comprehensive evaluation metrics:

```bash
# Evaluate a trained model
python main.py --config configs/baseline.yaml --phase test --load-weights ./work_dir/baseline_experiment/best_model.pt
```

Evaluation results include:
- Word Error Rate (WER)
- Sentence Error Rate (SER)
- Detailed error analysis

# checkpoint

| Model       | Download |
|-------------|----------|
|VAC          | 🧠 [Checkpoint]() |
|TLP          | 🧠 [Checkpoint]() |
|SEN          | 🧠 [Checkpoint]() |
|CorrNet      | 🧠 [Checkpoint]() |
|SlowFast     | 🧠 [Checkpoint]() |


## Performance on Phoenix14

|model                        | dev wer          |test wer   |  
|-----------------------------|------------------|-----------|    
| VAC + SMKD                  | 19.9 ↑           | 21.3 ↑    |  
| TLP                         | 20.2 ↑           | 20.8 -    |  
| SEN                         | 19.9 ↑           | 19.8 ↓    |  
| CorrNet                     | 20.2 ↑           | 20.6 ↑    |  
| SlowFast                    | 21.8 ↑           | 21.5 ↑    |  


## Experiment Management

OpenSLR integrates with Weights & Biases for experiment tracking:

```yaml
# Enable Weights & Biases in your config file
wandb:
  enable: true
  project: "openslr"
  entity: "your_wandb_username"
```

## API Reference

For detailed API documentation, please refer to the [online documentation](https://yourusername.github.io/OpenSLR/).

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is inspired by various state-of-the-art sign language recognition research
- We thank the contributors to the open-source libraries used in this project

## Citation

If you use this framework in your research, please cite:

```
@misc{openslr2024,
  title={OpenSLR: An Open Sign Language Recognition Framework},
  author={Your Name and Collaborators},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/yourusername/OpenSLR}},
}
```

## Contact

For questions or support, please open an issue on GitHub or contact the project maintainers.
