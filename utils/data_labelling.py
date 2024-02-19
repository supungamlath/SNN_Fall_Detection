import os
import pandas as pd


def label_URFD_dataset():
    dataset_folder = f"{os.environ['root_folder']}/data/urfd-spiking-dataset-240"
    folder_names = [
        folder
        for folder in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, folder))
    ]
    labels = [1 if "fall" in name else 0 for name in folder_names]
    df = pd.DataFrame({"folder_name": folder_names, "label": labels})
    df.to_csv(f"{dataset_folder}/labels.csv", index=False)


def label_HAR_UP_dataset():
    dataset_folder = f"{os.environ['root_folder']}/data/har-up-spiking-dataset-240"
    folder_names = [
        folder
        for folder in os.listdir(dataset_folder)
        if os.path.isdir(os.path.join(dataset_folder, folder))
    ]
    labels = [1] * len(folder_names)
    df = pd.DataFrame({"folder_name": folder_names, "label": labels})
    df.to_csv(f"{dataset_folder}/labels.csv", index=False)


# label_URFD_dataset()
# label_HAR_UP_dataset()
