import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import h5py


class SpikingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        max_time=15.0,
        nb_steps=1000,
        read_csv=True,
    ):
        self.root_dir = root_dir
        self.max_time = max_time
        self.nb_steps = nb_steps
        self.frame_height = 180
        self.frame_width = 240
        self.max_timestamp = max_time * 1e6
        self.nb_pixels = self.frame_height * self.frame_width

        if read_csv:
            # Load the CSV file into a DataFrame
            labels_file = os.path.join(root_dir, "labels.csv")
            if os.path.exists(labels_file):

                df = pd.read_csv(labels_file)
                self.folder_names = df["folder_name"].tolist()
                self.labels = df["label"].tolist()
            else:
                self.folder_names = [
                    folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))
                ]
                self.labels = [1] * len(self.folder_names)

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        events_file = self.folder_names[idx] + ".h5"
        try:
            spike_data = h5py.File(os.path.join(self.root_dir, self.folder_names[idx], events_file), "r")
            spike_tuples = np.array(spike_data["events"])
            # Remove events that occur after max_timestamp
            spike_tuples = spike_tuples[spike_tuples[:, 0] < self.max_timestamp]
            sample = {
                "timestamp": spike_tuples[:, 0] / 1e6,
                "x": spike_tuples[:, 1],
                "y": spike_tuples[:, 2],
                "polarity": spike_tuples[:, 3],
            }
            return sample, self.labels[idx]
        except Exception as e:
            print(f"Error reading {events_file}: {e}")
            return None, None

    def edit_label(self, idx, label):
        self.labels[idx] = label

    def get_folder_name(self, idx):
        return self.folder_names[idx]

    def get_label(self, idx):
        return self.labels[idx]

    def get_video_path(self, idx):
        video_path = os.path.join(self.root_dir, self.folder_names[idx], "dvs-video.avi")
        return os.path.normpath(video_path)

    def random_split(self, test_size=0.25, shuffle=True, batch_size=7):
        train_dataset = SpikingDataset(
            root_dir=self.root_dir,
            max_time=self.max_time,
            nb_steps=self.nb_steps,
            read_csv=False,
        )
        test_dataset = SpikingDataset(
            root_dir=self.root_dir,
            max_time=self.max_time,
            nb_steps=self.nb_steps,
            read_csv=False,
        )

        # Ensure reproducibility
        np.random.seed(42)

        # Combine folder names and labels
        combined = list(zip(self.folder_names, self.labels))

        # Shuffle combined data if required
        if shuffle:
            np.random.shuffle(combined)

        # Calculate total number of elements
        total_elements = len(combined)

        # Calculate desired sizes
        test_size_elements = int(total_elements * test_size)
        train_size_elements = total_elements - test_size_elements

        # Adjust sizes to be divisible by batch_size
        test_size_adjusted = (test_size_elements // batch_size) * batch_size
        train_size_adjusted = (train_size_elements // batch_size) * batch_size

        # Split the data
        train_data = combined[0:train_size_adjusted]
        test_data = combined[train_size_adjusted : train_size_adjusted + test_size_adjusted]

        # Separate the folder names and labels
        train_dataset.folder_names, train_dataset.labels = zip(*train_data)
        test_dataset.folder_names, test_dataset.labels = zip(*test_data)

        return train_dataset, test_dataset

    def save_labels(self):
        df = pd.DataFrame(
            {
                "folder_name": self.folder_names,
                "label": self.labels,
            }
        )
        df.to_csv(
            os.path.join(self.root_dir, "labels.csv"),
            index=False,
        )
