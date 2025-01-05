import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import h5py


class SpikingDataset(Dataset):
    def __init__(
        self,
        root_dir,
        time_duration=60.0,
        read_csv=True,
    ):
        self.root_dir = root_dir
        self.time_duration = time_duration
        self.frame_height = 180
        self.frame_width = 240
        self.max_timestamp = time_duration * 1e6
        self.nb_pixels = self.frame_height * self.frame_width
        self.scaling_factor = 1.57

        if read_csv:
            labels = self.get_fall_flags()
            self.folder_names = list(labels.keys())
            self.labels = list(labels.values())

    def get_fall_flags(self):
        """
        Reads the labels.csv file and extracts the 60 fall flag values for each video.

        Returns:
            dict: A dictionary where keys are video names and values are lists of fall flags for each second.
        """
        # Read the CSV file
        labels_file = os.path.join(self.root_dir, "labels.csv")
        df = pd.read_csv(labels_file)

        # Initialize the result dictionary
        fall_flags_dict = {}

        # Iterate over each row
        for _, row in df.iterrows():
            # Extract video name and fall flags
            video_name = row["name"]
            fall_flags = [row[f"second_{i+1}"] for i in range(60)]

            # Add to the dictionary
            fall_flags_dict[video_name] = fall_flags

        return fall_flags_dict

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
                "timestamp": spike_tuples[:, 0] / 1e6 * self.scaling_factor,
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

    def _adjust_to_batch_size(self, data, batch_size=7):
        size = len(data)
        adjusted_size = (size // batch_size) * batch_size
        return data[:adjusted_size]

    def random_split(self, test_size=0.25, shuffle=True):
        train_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            read_csv=False,
        )
        test_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            read_csv=False,
        )

        # Ensure reproducibility
        np.random.seed(42)

        # Combine folder names and labels
        combined = list(zip(self.folder_names, self.labels))

        # Shuffle combined data if required
        if shuffle:
            np.random.shuffle(combined)

        # Calculate the split index
        total_elements = len(combined)
        test_size_elements = int(total_elements * test_size)

        # Split the data
        train_data = combined[:-test_size_elements]
        test_data = combined[-test_size_elements:]

        # Adjust data sizes to be divisible by batch_size
        train_data = self._adjust_to_batch_size(train_data)
        test_data = self._adjust_to_batch_size(test_data)

        # Separate folder names and labels
        train_dataset.folder_names, train_dataset.labels = zip(*train_data) if train_data else ([], [])
        test_dataset.folder_names, test_dataset.labels = zip(*test_data) if test_data else ([], [])

        return train_dataset, test_dataset

    def split_by_subjects(self):
        train_subjects = {
            "Subject1",
            "Subject3",
            "Subject4",
            "Subject7",
            "Subject10",
            "Subject11",
            "Subject12",
            "Subject13",
            "Subject14",
        }
        test_subjects = {"Subject15", "Subject16", "Subject17"}
        dev_subjects = {"Subject2", "Subject5", "Subject6", "Subject8", "Subject9"}

        train_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            read_csv=False,
        )
        dev_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            read_csv=False,
        )
        test_dataset = SpikingDataset(
            root_dir=self.root_dir,
            time_duration=self.time_duration,
            read_csv=False,
        )

        # Combine folder names and labels
        combined = list(zip(self.folder_names, self.labels))

        # Separate data based on subjects
        train_data = []
        dev_data = []
        test_data = []

        for folder_name, label in combined:
            subject = folder_name.split("Activity")[0]
            if subject in train_subjects:
                train_data.append((folder_name, label))
            elif subject in test_subjects:
                test_data.append((folder_name, label))
            elif subject in dev_subjects:
                dev_data.append((folder_name, label))

        train_data = self._adjust_to_batch_size(train_data)
        dev_data = self._adjust_to_batch_size(dev_data)
        test_data = self._adjust_to_batch_size(test_data)

        # Separate the folder names and labels
        train_dataset.folder_names, train_dataset.labels = zip(*train_data) if train_data else ([], [])
        dev_dataset.folder_names, dev_dataset.labels = zip(*dev_data) if dev_data else ([], [])
        test_dataset.folder_names, test_dataset.labels = zip(*test_data) if test_data else ([], [])

        return train_dataset, dev_dataset, test_dataset

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
