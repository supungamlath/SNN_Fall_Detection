# Define the path to the video datasets
import json
import os
import pandas as pd


def get_datasets_dirs():
    datasets_dirs = {
        # "UR Fall Dataset": f"{os.environ['root_folder']}/data/urfd-spiking-dataset-240",
        "HAR UP Fall Dataset": f"{os.environ['root_folder']}/data/har-up-spiking-dataset-240",
    }
    return datasets_dirs


# Function to save parameters to file
def save_params(file_dir, params):
    with open(file_dir, "w") as f:
        json.dump(params, f, indent=4)


# Function to load parameters from file
def load_params(file_dir):
    if os.path.exists(file_dir):
        with open(file_dir, "r") as f:
            params = json.load(f)
        return params
    else:
        return {}


# Convert training results JSON data to a Pandas DataFrame
def training_json_to_dataframe(data):
    records = []
    for model_name, experiments in data.items():
        for experiment in experiments:
            record = {
                "model_name": model_name,
                "datetime": experiment.get("datetime"),
                "dataset": experiment.get("dataset"),
                "train_test_ratio": experiment.get("train_test_ratio"),
                "nb_epochs": experiment.get("nb_epochs"),
                "learning_rate": experiment.get("learning_rate"),
            }
            if "train_metrics_hist" in experiment:
                record["train_accuracy_hist"] = [metrics["accuracy"] for metrics in experiment["train_metrics_hist"]]
                record["train_precision_hist"] = [metrics["precision"] for metrics in experiment["train_metrics_hist"]]
                record["train_recall_hist"] = [metrics["recall"] for metrics in experiment["train_metrics_hist"]]
                record["train_f1_score_hist"] = [metrics["f1_score"] for metrics in experiment["train_metrics_hist"]]
                record["train_loss_hist"] = [metrics["loss"] for metrics in experiment["train_metrics_hist"]]
            if "dev_metrics_hist" in experiment:
                record["test_accuracy_hist"] = [metrics["accuracy"] for metrics in experiment["dev_metrics_hist"]]
                record["test_precision_hist"] = [metrics["precision"] for metrics in experiment["dev_metrics_hist"]]
                record["test_recall_hist"] = [metrics["recall"] for metrics in experiment["dev_metrics_hist"]]
                record["test_f1_score_hist"] = [metrics["f1_score"] for metrics in experiment["dev_metrics_hist"]]
            records.append(record)
    df = pd.DataFrame(records)
    return df


# Convert models information JSON data to a Pandas DataFrame
def models_info_json_to_dataframe(data):
    records = []
    for model_name, info in data.items():
        record = {"model_name": model_name}
        record.update(info)
        records.append(record)
    df = pd.DataFrame(records)
    return df


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                return True
        return False
