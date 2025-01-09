import os
from pathlib import Path

import configparser
from clearml import Dataset, Task

from models.SpikingNN import SpikingNN
from utils.SpikingDataset import SpikingDataset
from utils.SpikingDataLoader import SpikingDataLoader
from utils.Trainer import Trainer
from utils.clearml_helpers import report_metrics


def main():
    # Init ClearML project
    task = Task.init(
        project_name="NeuroFall", task_name="model_training", output_uri=True, auto_resource_monitoring=False
    )

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
        "use_regularizer": config.getboolean("TRAINING", "use_regularizer"),
        "early_stopping": config.getboolean("TRAINING", "early_stopping"),
    }
    task.connect(training_params, name="Training Parameters")

    # Read dataset parameters
    dataset_params = {
        "time_duration": float(config["DATASET"]["time_duration"]),
        "bias_ratio": float(config["DATASET"]["bias_ratio"]),
        "camera1_only": config.getboolean("DATASET", "camera1_only"),
        "split_by": config["DATASET"]["split_by"],
    }
    task.connect(dataset_params, name="Dataset Parameters")

    # Define folder and paths
    root_folder = Path(config["DEFAULT"]["root_dir"] or os.getcwd())
    dataset_dir = root_folder / config["DATASET"]["data_dir"]
    model_dir = root_folder / config["MODEL"]["save_dir"]
    model_save_file = model_dir / f"{model_name}.pth"

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
        time_duration=dataset_params["time_duration"],
        camera1_only=dataset_params["camera1_only"],
    )

    # Splitting the dataset
    if dataset_params["split_by"] == "subjects":
        train_dataset, dev_dataset, test_dataset = dataset.split_by_subjects(batch_size=training_params["batch_size"])
    elif dataset_params["split_by"] == "trials":
        train_dataset, dev_dataset, test_dataset = dataset.split_by_trials(batch_size=training_params["batch_size"])
    else:
        raise ValueError("Invalid value for split_by parameter")

    model = SpikingNN(
        layer_sizes=[dataset.nb_pixels] + model_params["hidden_layers"] + [2],
        nb_steps=model_params["nb_steps"],
        time_step=dataset_params["time_duration"] / model_params["nb_steps"],
        tau_mem=model_params["tau_mem"] * 1e-3,
        tau_syn=model_params["tau_syn"] * 1e-3,
    )

    # Creating DataLoader instances
    train_loader = SpikingDataLoader(
        dataset=train_dataset,
        nb_steps=model_params["nb_steps"],
        batch_size=training_params["batch_size"],
        shuffle=False,
    )
    dev_loader = SpikingDataLoader(
        dataset=dev_dataset, nb_steps=model_params["nb_steps"], batch_size=training_params["batch_size"], shuffle=False
    )
    test_loader = SpikingDataLoader(
        dataset=test_dataset, nb_steps=model_params["nb_steps"], batch_size=training_params["batch_size"], shuffle=False
    )

    def evaluate_epoch_callback(dev_metrics, epoch):
        report_metrics("dev", dev_metrics, epoch + 1)
        print(f"Saved dev record for epoch {epoch + 1}")

    def training_epoch_callback(train_metrics, epoch):
        if epoch % 3 == 0:
            model.save(model_save_file)
        report_metrics("train", train_metrics, epoch + 1)
        print(f"Saved train record for epoch {epoch + 1}")

    # Train the model
    trainer = Trainer(model=model)
    trainer.train(
        train_loader,
        evaluate_dataloader=dev_loader,
        nb_epochs=training_params["nb_epochs"],
        lr=training_params["learning_rate"],
        use_regularizer=training_params["use_regularizer"],
        regularizer_alpha=training_params["reg_alpha"],
        step_lr_size=training_params["step_lr_size"],
        step_lr_gamma=training_params["step_lr_gamma"],
        stop_early=training_params["early_stopping"],
        dataset_bias_ratio=dataset_params["bias_ratio"],
        evaluate_callback=evaluate_epoch_callback,
        train_callback=training_epoch_callback,
    )

    # Save the model
    model.save(model_save_file)

    # Test the model
    test_metrics_dict = trainer.test(test_loader)
    report_metrics("test", test_metrics_dict, 0)


if __name__ == "__main__":
    main()
