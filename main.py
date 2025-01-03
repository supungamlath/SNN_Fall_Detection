from datetime import datetime
import torch
import configparser

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

# Read training parameters
learning_rate = float(config["TRAINING"]["learning_rate"])
nb_epochs = int(config["TRAINING"]["nb_epochs"])
batch_size = int(config["TRAINING"]["batch_size"])

# Read dataset parameters
max_time = float(config["DATASET"]["max_time"])
nb_steps = int(config["DATASET"]["nb_steps"])

# Define root folder and paths
root_folder = config["DEFAULT"]["root_dir"]
training_runs_file = f"{root_folder}{config['TRAINING']['save_file']}"
dataset_dir = f"{root_folder}{config['DATASET']['data_dir']}"
model_dir = f"{root_folder}{config['MODEL']['save_dir']}"

# Load dataset
dataset = SpikingDataset(
    root_dir=dataset_dir,
    max_time=max_time,
    nb_steps=nb_steps,
)

# Splitting the dataset
train_dataset, dev_dataset, test_dataset = dataset.split_by_subjects()

# Creating the model
model = SpikingNN(
    layer_sizes=[dataset.nb_pixels, 2000, 2],
    nb_steps=dataset.nb_steps,
)
model = torch.compile(model)

# Creating DataLoader instances
train_loader = SpikingDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
dev_loader = SpikingDataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = SpikingDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the model
# model = SpikingNN.load(f"{root_folder}/models/saved/model_v5.pth")
# model.eval()

# Load previous training runs
training_params = load_params(training_runs_file)

if model_name not in training_params:
    training_params[model_name] = []
training_params[model_name].append(
    {
        "datetime": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "dataset": config["DATASET"]["name"],
        "nb_epochs": nb_epochs,
        "learning_rate": learning_rate,
    }
)
save_params(training_runs_file, training_params)

# Train the model
trainer = Trainer(model=model)
train_metrics_hist, dev_metrics_hist = trainer.train(
    train_loader,
    nb_epochs=nb_epochs,
    lr=learning_rate,
    evaluate_dataloader=dev_loader,
    stop_early=True,
)
training_params[model_name][-1]["train_metrics_hist"] = train_metrics_hist
training_params[model_name][-1]["dev_metrics_hist"] = dev_metrics_hist
save_params(training_runs_file, training_params)

# Test the model
test_metrics_dict = trainer.test(test_loader)

# Save the model
model.save(f"{model_dir}/{model_name}.pth")

# Visualize the output spikes
# trainer.visualize_output(test_loader, 1)
