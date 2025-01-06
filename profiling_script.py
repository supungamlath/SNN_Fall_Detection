import configparser
import torch

from models.SpikingNN import SpikingNN
from utils.SpikingDataLoader import SpikingDataLoader
from utils.SpikingDataset import SpikingDataset

# Load configuration
config = configparser.ConfigParser()
config.read("config.txt")

model_params = {
    "hidden_layers": list(map(int, config["MODEL"]["hidden_layers"].split(","))),
    "tau_mem": float(config["MODEL"]["tau_mem"]),
    "tau_syn": float(config["MODEL"]["tau_syn"]),
    "nb_steps": int(config["MODEL"]["nb_steps"]),
}

# Read training parameters
training_params = {
    "learning_rate": float(config["TRAINING"]["learning_rate"]),
    "reg_alpha": float(config["TRAINING"]["reg_alpha"]),
    "step_lr_size": int(config["TRAINING"]["step_lr_size"]),
    "step_lr_gamma": float(config["TRAINING"]["step_lr_gamma"]),
    "nb_epochs": int(config["TRAINING"]["nb_epochs"]),
    "batch_size": int(config["TRAINING"]["batch_size"]),
}

# Read dataset parameters
time_duration = float(config["DATASET"]["time_duration"])

# Load dataset
dataset = SpikingDataset(
    root_dir="data/har-up-spiking-dataset-240",
    time_duration=60.0,
)

# Splitting the dataset
train_dataset, dev_dataset, test_dataset = dataset.split_by_subjects()

# Creating DataLoader instances
train_loader = SpikingDataLoader(
    dataset=train_dataset, nb_steps=model_params["nb_steps"], batch_size=training_params["batch_size"], shuffle=False
)
dev_loader = SpikingDataLoader(
    dataset=dev_dataset, nb_steps=model_params["nb_steps"], batch_size=training_params["batch_size"], shuffle=False
)
test_loader = SpikingDataLoader(
    dataset=test_dataset, nb_steps=model_params["nb_steps"], batch_size=training_params["batch_size"], shuffle=False
)

model = SpikingNN(
        layer_sizes=[dataset.nb_pixels] + model_params["hidden_layers"] + [2],
        nb_steps=model_params["nb_steps"],
        time_step=time_duration / model_params["nb_steps"],
        tau_mem=model_params["tau_mem"] * 1e-3,
        tau_syn=model_params["tau_syn"] * 1e-3,
    )

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=3, repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
    record_shapes=False,
    with_stack=True,
    profile_memory=False,
    with_flops=False,
) as prof:
    # Training Loop
    for step, batch_data in enumerate(train_loader):
        x_local, y_local = batch_data
        prof.step()
        if step >= 1 + 1 + 3:
            break
        model.forward(x_local.to_dense())

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

