import os
from pathlib import Path

import configparser
from datetime import datetime
from clearml import Dataset, Task

from models.SpikingNN import SpikingNN
from utils.SpikingDataset import SpikingDataset
from utils.SpikingDataLoader import SpikingDataLoader
from utils.Trainer import Trainer
from utils.clearml_helpers import report_metrics
from utils.helpers import load_params, save_params


def main():
    # Init ClearML project
    task = Task.init(project_name="NeuroFall", task_name="model_training", output_uri=True)

    # Load configuration
    config = configparser.ConfigParser()
    config.read("config.txt")

    # Read model parameters
    model_name = config["MODEL"]["name"]
    model_params = {
        "hidden_layers": list(map(int, config["MODEL"]["hidden_layers"].split(","))),
        "tau_mem": float(config["MODEL"]["tau_mem"]),
        "tau_syn": float(config["MODEL"]["tau_syn"]),
        "nb_steps": int(config["MODEL"]["nb_steps"]),
    }
    task.connect(model_params, name="Model Parameters")

    # Read training parameters
    training_params = {
        "learning_rate": float(config["TRAINING"]["learning_rate"]),
        "reg_alpha": float(config["TRAINING"]["reg_alpha"]),
        "step_lr_size": int(config["TRAINING"]["step_lr_size"]),
        "step_lr_gamma": float(config["TRAINING"]["step_lr_gamma"]),
        "nb_epochs": int(config["TRAINING"]["nb_epochs"]),
        "batch_size": int(config["TRAINING"]["batch_size"]),
    }
    task.connect(training_params, name="Training Parameters")

    # Read dataset parameters
    time_duration = float(config["DATASET"]["time_duration"])

    # Define folder and paths
    root_folder = Path(config["DEFAULT"]["root_dir"] or os.getcwd())
    dataset_dir = root_folder / config["DATASET"]["data_dir"]
    model_dir = root_folder / config["MODEL"]["save_dir"]
    model_save_file = model_dir / f"{model_name}.pth"
    models_records_file = root_folder / config["MODEL"]["save_file"]
    training_records_file = root_folder / config["TRAINING"]["save_file"]
    training_logs_file = root_folder / config["TRAINING"]["logs_file"]

    # Create directories if they don't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset if it doesn't exist
    if not dataset_dir.exists():
        dataset_dir = Dataset.get(
            dataset_name="har-up-spiking-dataset-240-min", alias="HAR UP Fall Dataset"
        ).get_local_copy()

    # Load dataset
    dataset = SpikingDataset(
        root_dir=dataset_dir,
        time_duration=time_duration,
    )

    # Splitting the dataset
    train_dataset, dev_dataset, test_dataset = dataset.split_by_subjects(batch_size=training_params["batch_size"])

    # Load existing model parameters
    model_records = load_params(models_records_file)

    # Creating the model
    model_records[model_name] = {
        "snn_layers": [dataset.nb_pixels] + model_params["hidden_layers"] + [2],
        "nb_steps": model_params["nb_steps"],
        "time_step": time_duration / model_params["nb_steps"],
        "tau_mem": model_params["tau_mem"] * 1e-3,
        "tau_syn": model_params["tau_syn"] * 1e-3,
        "time_duration": time_duration,
    }

    # Load model if it exists
    if os.path.exists(model_save_file):
        model = SpikingNN.load(model_save_file)
    else:
        model = SpikingNN(
            layer_sizes=[dataset.nb_pixels] + model_params["hidden_layers"] + [2],
            nb_steps=model_params["nb_steps"],
            time_step=time_duration / model_params["nb_steps"],
            tau_mem=model_params["tau_mem"] * 1e-3,
            tau_syn=model_params["tau_syn"] * 1e-3,
        )
    save_params(models_records_file, model_records)

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

    # Load training record
    training_records = load_params(training_records_file)

    if model_name not in training_records:
        training_records[model_name] = []
    training_records[model_name].append(
        {
            "datetime": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "dataset": config["DATASET"]["name"],
            "nb_epochs": training_params["nb_epochs"],
            "learning_rate": training_params["learning_rate"],
        }
    )
    save_params(training_records_file, training_records)


    def save_training_epoch_callback(train_metrics_hist, dev_metrics_hist):
        training_records[model_name][-1]["train_metrics_hist"] = train_metrics_hist
        training_records[model_name][-1]["dev_metrics_hist"] = dev_metrics_hist
        save_params(training_records_file, training_records)
        model.save(model_save_file)
        report_metrics("train", train_metrics_hist[-1], len(train_metrics_hist))
        report_metrics("dev", dev_metrics_hist[-1], len(train_metrics_hist))
        print(f"Saved record for epoch {len(train_metrics_hist)}")


    # Train the model
    trainer = Trainer(model=model)
    train_metrics_hist, dev_metrics_hist = trainer.train(
        train_loader,
        nb_epochs=training_params["nb_epochs"],
        lr=training_params["learning_rate"],
        reg_alpha=training_params["reg_alpha"],
        step_lr_size=training_params["step_lr_size"],
        step_lr_gamma=training_params["step_lr_gamma"],
        evaluate_dataloader=dev_loader,
        stop_early=True,
        callback_fn=save_training_epoch_callback,
    )

    # Test the model
    test_metrics_dict = trainer.test(test_loader)
    training_records[model_name][-1]["test_metrics"] = test_metrics_dict
    save_params(training_records_file, training_records)
    report_metrics("test", test_metrics_dict, len(train_metrics_hist))


if __name__ == '__main__':
    main()