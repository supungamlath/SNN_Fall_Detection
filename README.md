# Spiking Neural Network (SNN) Project

This project implements a Spiking Neural Network (SNN) for processing spiking datasets. It uses ClearML for experiment tracking and supports multiple SNN architectures, including custom implementations and pre-defined models like `SNNTorchConv`, `SNNTorchLeaky`, and `SNNTorchSyn`.

## Project Structure

```
.
├── main.py                 # Entry point for the project
├── config.txt              # Configuration file for model, training, and dataset parameters
├── models/                 # Contains SNN model implementations
│   ├── SNNTorchConv.py
│   ├── SNNTorchLeaky.py
│   ├── SNNTorchSyn.py
│   ├── SpikingNN.py
│   └── saved/              # Directory for saving trained models
├── utils/                  # Utility scripts
│   ├── BinaryTrainer.py
│   ├── MultiTrainer.py
│   ├── SpikingDataset.py
│   ├── SpikingDataLoader.py
│   └── clearml_helpers.py
├── data/                   # Directory for datasets
├── notebooks/              # Jupyter notebooks for experimentation
└── requirements.txt        # Python dependencies
```

## Features

- **Customizable SNN Architectures**: Easily switch between different SNN models.
- **Dataset Handling**: Supports spiking datasets with configurable splitting by subjects or trials.
- **Training and Evaluation**: Includes training, validation, and testing pipelines with ClearML integration.
- **Experiment Tracking**: Automatically logs parameters and metrics to ClearML.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/supungamlath/SNN_Fall_Detection
   cd SNN_Fall_Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure ClearML:
   - Update the `data/clearml.conf` file with your ClearML credentials.

## Configuration

Modify the `config.txt` file to set parameters for the model, training, and dataset. Example:

```ini
[DEFAULT]
root_dir = ./

[MODEL]
name = SpikingNN
hidden_layers = 128,64
tau_mem = 20.0
tau_syn = 5.0
nb_steps = 100
multiclass = True
save_dir = ./models/saved

[TRAINING]
learning_rate = 0.001
reg_alpha = 0.01
nb_epochs = 50
batch_size = 32
use_regularizer = True
early_stopping = True

[DATASET]
time_duration = 1.0
bias_ratio = 0.5
camera1_only = False
split_by = subjects
data_dir = ./har-up-spiking-dataset-240
```

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. The script will:
   - Load the configuration.
   - Initialize the ClearML task.
   - Download the dataset if not already available.
   - Train the selected SNN model.
   - Save the trained model.
   - Evaluate the model on the test set.

## Notebooks

Explore the `notebooks/` directory for Jupyter notebooks that demonstrate specific tasks, such as dataset labeling and converting videos to events.

## ClearML

This project uses ClearML for experiment tracking and saving model checkpoints. If you have a ClearML server set up, place the configuration in `data/clearml.conf`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [ClearML](https://clear.ml/) for experiment tracking and saving model checkpoints.
- [SNNTorch](https://snntorch.readthedocs.io/) for spiking neural network utilities.
