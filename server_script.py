import argparse
import zipfile

import torch
import boto3
import gdown
import logging
import os
import sys
from models.SpikingNN import SpikingNN
from sagemaker_training import environment
from utils.SpikingDataset import SpikingDataset
from utils.SpikingDataLoader import SpikingDataLoader
from utils.Trainer import Trainer

env = environment.Environment()
job_name = env["job_name"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def upload_to_s3(
    path,
    s3_path,
    bucket_name="ai-ml-training-models",
):
    s3 = boto3.client("s3")
    s3.upload_file(path, bucket_name, s3_path)
    logger.info(f"Finished uploading to {bucket_name}")


def download_from_gdrive(dataset_dir, file_id):
    # dataset_dir = "../data/urfd-spiking-dataset-240"
    # Check if it exists
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

        # Destination filename
        destination = f"{dataset_dir}.zip"
        # Download URL
        url = f"https://drive.google.com/uc?id={file_id}"

        # Download the file from Google Drive
        gdown.download(url, destination, quiet=False)

        # Extract the zip file
        with zipfile.ZipFile(destination, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

        # Remove the zip file
        os.remove(destination)
        print(f"Dataset extracted to {dataset_dir}")


if __name__ == "__main__":
    # Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--reg-alpha", type=float, default=2e-6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--hidden-layers", type=list, nargs="+", default=[2000])

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    # Parse Parameters
    args = parser.parse_args()

    dataset_dir = f"{args.data_dir}/har-up-spiking-dataset-240"
    # Download the dataset
    download_from_gdrive(
        dataset_dir=dataset_dir,
        file_id="1NtUqy2ofIkHYgfloJXgY06nlU39Jjcp0",
    )
    dataset = SpikingDataset(
        root_dir=dataset_dir,
        time_duration=15.0,
        nb_steps=1000,
    )

    # Splitting the dataset
    train_dataset, test_dataset = dataset.random_split(test_size=args.test_size, shuffle=True)

    layer_sizes = [dataset.nb_pixels] + args.hidden_layers + [2]
    model = SpikingNN(
        layer_sizes=layer_sizes,
        nb_steps=dataset.nb_steps,
    )
    model = torch.compile(model)

    # Creating DataLoader instances
    train_loader = SpikingDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = SpikingDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train the model
    trainer = Trainer(model=model)
    trainer.train(
        train_loader,
        evaluate_dataloader=test_loader,
        nb_epochs=args.epochs,
        lr=args.learning_rate,
        regularizer_alpha=args.reg_alpha,
    )

    # Save the model
    model_save_dir = f"{args.model_dir}/{job_name}.pth"
    model.save(model_save_dir)
    upload_to_s3(model_save_dir, s3_path=f"GTIN_Anomaly/SNN_Models/{job_name}")
