# OpenSLR Core

This directory contains the core implementation of the OpenSLR (Open Sign Language Recognition) framework.

## Core Components

### 1. Main Entry Point (`main.py`)

The main program entry point that initializes all managers and starts the training/evaluation process:

```python
# Example usage
python main.py --config configs/baseline.yaml --work-dir ./work_dir/baseline_experiment
```

### 2. Manager System

The manager system coordinates different components of the framework:

- **ArgumentManager**: Handles command-line arguments
- **ConfigManager**: Manages configuration files and settings
- **ExperimentManager**: Orchestrates the training/evaluation process
- **DatasetManager**: Manages dataset loading and preprocessing
- **ModelManager**: Handles model creation and loading

### 3. Model Architectures

The models directory contains various sign language recognition architectures:

#### Model Definitions
- `build_function.py`: Factory functions to create different model architectures
- `modules/`: Shared model components
- `senmodules/`: Specialized modules for sign language recognition

#### Supported Architectures
- SlowFast
- TLP (Two-Stream Lightweight Pyramid)
- VAC (Visual Attention Consistency)
- CorrNet

### 4. Dataset System

The dataset component handles data loading and preprocessing:

- **VideoDataset**: Main dataset class for loading video data
- **Dataloader**: Efficient data loading with memory mapping support
- **Preprocessing**: Scripts for data preparation and augmentation

### 5. Configuration System

Configuration files define experiment parameters:

- **Baseline Configs**: Reference configurations for different models
- **Custom Configs**: Easily create custom experiment configurations

### 6. Pipeline

The pipeline component orchestrates the training/evaluation workflow:

- **Training Pipeline**: Handles the training loop, loss calculation, and optimization
- **Evaluation Pipeline**: Manages model evaluation and metric calculation

## Usage

### Training

```bash
# Train with a baseline configuration
python main.py --config configs/baseline.yaml --work-dir ./work_dir/baseline_experiment

# Train with a custom configuration
python main.py --config configs/my_experiment.yaml --work-dir ./work_dir/my_experiment
```

### Evaluation

```bash
# Evaluate a trained model
python main.py --config configs/baseline.yaml --phase test --load-weights ./work_dir/baseline_experiment/best_model.pt
```

### Inference

```python
from OpenSLR import infer

# Load video data
video_data = ...  # Shape: [batch, channels, frames, height, width]
video_length = ...  # Sequence length

# Perform inference
recognized_sents = infer(video_data, video_length)
print("Recognition Result:", recognized_sents)
```

## Configuration Files

Configuration files define all experiment parameters:

```yaml
# Example configuration snippet
feeder: dataset.dataloader_video.VideoDataset
phase: train
dataset: phoenix2014
num_epoch: 80
work_dir: ./work_dir/baseline/
batch_size: 8

model: models.build_function.build_slowfast
model_args:
    num_classes: 1296
    hidden_size: 1024
    c2d_type: slowfast101
```

### Key Configuration Sections

- **feeder**: Dataset loader configuration
- **model**: Model architecture and parameters
- **optimizer_args**: Optimization settings
- **device**: GPU devices to use
- **log_interval**: Logging frequency
- **eval_interval**: Evaluation frequency
- **save_interval**: Model saving frequency
- **wandb**: Weights & Biases integration settings

## Dataset Preparation

Before training, you need to prepare your dataset:

```bash
# Preprocess the Phoenix2014 dataset
cd preprocess
python dataset_preprocess.py --dataset phoenix2014 --dataset-root /path/to/phoenix2014
```

## Adding Custom Models

To add a new model architecture:

1. Create the model class in `models/modules/`
2. Add a build function in `models/build_function.py`
3. Update the configuration file to use your new model

## Extending Datasets

To add support for a new dataset:

1. Create a dataset class in `dataset/`
2. Implement the required methods: `__init__`, `__len__`, `__getitem__`
3. Update the configuration file to use your new dataset

## Experiment Tracking

OpenSLR integrates with Weights & Biases for experiment tracking:

1. Enable W&B in your configuration file
2. Set up your W&B account
3. Track experiments, metrics, and visualizations in the W&B dashboard

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size in the configuration file
   - Enable memory mapping with `datatype: 'memmap'`

2. **Data Loading Issues**:
   - Check dataset paths in the configuration
   - Verify dataset preprocessing was completed correctly

3. **GPU Issues**:
   - Check CUDA installation
   - Verify GPU availability with `torch.cuda.is_available()`
   - Adjust `device` parameter in the configuration file

## Development

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Testing

```bash
# Run unit tests
python -m unittest discover
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact

For questions or support, please open an issue on GitHub.
