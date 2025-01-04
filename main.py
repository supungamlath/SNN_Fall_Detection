import os
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
import configparser
from datetime import datetime

from models.SpikingNN import SpikingNN
from utils.SpikingDataset import SpikingDataset
from utils.SpikingDataLoader import SpikingDataLoader
from utils.Trainer import Trainer
from utils.helpers import load_params, save_params

# Load configuration
config = configparser.ConfigParser()
config.read("config.txt")

# Read model parameters
model_name = config["MODEL"]["name"]
hidden_layers = list(map(int, config["MODEL"]["hidden_layers"].split(",")))
tau_mem = float(config["MODEL"]["tau_mem"])
tau_syn = float(config["MODEL"]["tau_syn"])

# Read training parameters
learning_rate = float(config["TRAINING"]["learning_rate"])
nb_epochs = int(config["TRAINING"]["nb_epochs"])
batch_size = int(config["TRAINING"]["batch_size"])

# Read dataset parameters
max_time = float(config["DATASET"]["max_time"])
nb_steps = int(config["DATASET"]["nb_steps"])

# Define folder and paths
root_folder = Path(config["DEFAULT"]["root_dir"] or os.getcwd())
dataset_dir = root_folder / config["DATASET"]["data_dir"]
model_dir = root_folder / config["MODEL"]["save_dir"]
model_save_file = model_dir / f"{model_name}.pth"
models_records_file = root_folder / config["MODEL"]["save_file"]
training_records_file = root_folder / config["TRAINING"]["save_file"]
training_logs_file = root_folder / config["TRAINING"]["logs_file"]

# Load dataset
dataset = SpikingDataset(
    root_dir=dataset_dir,
    max_time=max_time,
    nb_steps=nb_steps,
)

# Splitting the dataset
train_dataset, dev_dataset, test_dataset = dataset.split_by_subjects()

# Load existing model parameters
model_records = load_params(models_records_file)

# Creating the model
model_records[model_name] = {
    "snn_layers": [dataset.nb_pixels] + hidden_layers + [2],
    "nb_steps": nb_steps,
    "time_step": max_time / nb_steps,
    "tau_mem": tau_mem * 1e-3,
    "tau_syn": tau_syn * 1e-3,
    "max_time": max_time,
    "batch_size": batch_size,
}

# Load model if it exists
if os.path.exists(model_save_file):
    model = SpikingNN.load(model_save_file)
else:
    model = SpikingNN(
        layer_sizes=[dataset.nb_pixels] + hidden_layers + [2],
        nb_steps=nb_steps,
        time_step=max_time / nb_steps,
        tau_mem=tau_mem * 1e-3,
        tau_syn=tau_syn * 1e-3,
    )
# model = torch.compile(model)
save_params(models_records_file, model_records)

# Creating DataLoader instances
train_loader = SpikingDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
dev_loader = SpikingDataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = SpikingDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load training record
training_records = load_params(training_records_file)

if model_name not in training_records:
    training_records[model_name] = []
training_records[model_name].append(
    {
        "datetime": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "dataset": config["DATASET"]["name"],
        "nb_epochs": nb_epochs,
        "learning_rate": learning_rate,
    }
)
save_params(training_records_file, training_records)


def save_training_epoch_callback(train_metrics_hist, dev_metrics_hist):
    training_records[model_name][-1]["train_metrics_hist"] = train_metrics_hist
    training_records[model_name][-1]["dev_metrics_hist"] = dev_metrics_hist
    save_params(training_records_file, training_records)
    model.save(model_save_file)
    print(f"Saved record for epoch {len(train_metrics_hist)}")


# Train the model
trainer = Trainer(model=model)
with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as profiler:
    train_metrics_hist, dev_metrics_hist = trainer.train(
        train_loader,
        nb_epochs=nb_epochs,
        lr=learning_rate,
        evaluate_dataloader=dev_loader,
        stop_early=True,
        callback_fn=save_training_epoch_callback,
    )
    profiler.export_chrome_trace(training_logs_file)

# Test the model
test_metrics_dict = trainer.test(test_loader)
training_records[model_name][-1]["test_metrics"] = test_metrics_dict
save_params(training_records_file, training_records)
