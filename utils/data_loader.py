import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from utils.visualization import plot_voltage_traces


class EventsDataset:
    def __init__(
        self,
        datasets,
        device,
        frame_width=240,
        frame_height=180,
        max_time=10.0,
        nb_steps=1000,
        batch_size=7,
    ):
        self.datasets = datasets
        self.selected_dataset = None
        self.device = device

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.nb_pixels = frame_width * frame_height
        self.batch_size = batch_size
        self.max_time = max_time
        self.nb_steps = nb_steps

        self.folders = []
        self.video_files = []
        self.samples = []
        self.labels = []

        self.load_dataset()

    def __get_dataset_folder(self):
        if self.selected_dataset == "URFD":
            return "./data/urfd-spiking-dataset-240"
        elif self.selected_dataset == "HAR-UP":
            return "./data/har-up-spiking-dataset-240"

    def __get_label(self, name):
        if self.selected_dataset == "URFD":
            label = 1 if "fall" in name else 0
            return label
        elif self.selected_dataset == "HAR-UP":
            return 1

    def load_dataset(self):
        max_timestamp = 0
        for dataset in self.datasets:
            self.selected_dataset = dataset
            dataset_folder = self.__get_dataset_folder()

            folders = [
                folder
                for folder in os.listdir(dataset_folder)
                if os.path.isdir(os.path.join(dataset_folder, folder))
            ]
            print(f"Loading dataset from {dataset_folder}")
            for folder in folders:
                events_file = folder + ".h5"
                try:
                    spike_data = h5py.File(
                        os.path.join(dataset_folder, folder, events_file), "r"
                    )
                    spike_tuples = np.array(spike_data["events"])
                    spike_tuples = spike_tuples[
                        spike_tuples[:, 0] < self.max_time * 1e6
                    ]
                    sample = {
                        "timestamp": spike_tuples[:, 0] / 1e6,
                        "x": spike_tuples[:, 1],
                        "y": spike_tuples[:, 2],
                        "polarity": spike_tuples[:, 3],
                    }
                    sample_max = sample["timestamp"].max()
                    if sample_max > max_timestamp:
                        max_timestamp = sample_max
                    self.samples.append(sample)
                    self.labels.append(self.__get_label(folder))
                    self.video_files.append(f"{dataset_folder}/{folder}/dvs-video.avi")
                    self.folders.append(folder)

                except Exception as e:
                    print(f"Error in {events_file}")
                    print(e)

        print("Dataset loaded")
        print(f"No of samples: {len(self.samples)}")
        print(f"Max timestamp: {max_timestamp} s")

    def train_test_split(self, test_size=0.2, random_state=42):
        return train_test_split(
            self.samples, self.labels, test_size=test_size, random_state=random_state
        )

    def get_samples(self):
        return self.samples, self.labels

    def get_labels(self):
        return self.labels

    def get_video_files(self):
        return self.video_files

    def set_correct_label(self, index, label):
        self.labels[index] = label

    def save_labels(self):
        df = pd.DataFrame(
            {
                "folders": self.folders,
                "video_files": self.video_files,
                "label": self.labels,
            }
        )
        df.to_csv(
            f"{self.__get_dataset_folder(self.selected_dataset)}/labels.csv",
            index=False,
        )

    def load_labels(self):
        df = pd.read_csv(
            f"{self.__get_dataset_folder(self.selected_dataset)}/labels.csv"
        )
        self.folders = df["folders"].tolist()
        self.video_files = df["video_files"].tolist()
        self.labels = df["label"].tolist()

    def visualize_samples(self, nb_batches):
        batch_counter = 0

        for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
            self.samples,
            self.labels,
            self.batch_size,
            self.nb_steps,
            self.nb_pixels,
            self.max_time,
            self.device,
            shuffle=False,
        ):
            x_local_pixel_averaged = x_local.to_dense().mean(dim=2, keepdim=True)
            plot_voltage_traces(
                x_local_pixel_averaged.detach().cpu().numpy(),
                labels=y_local.detach().cpu().tolist(),
                dim=(1, self.batch_size),
            )

            batch_counter += 1
            if batch_counter == nb_batches:
                break


def sparse_data_generator_from_hdf5_spikes(
    X,
    y,
    batch_size,
    nb_steps,
    nb_units,
    max_time,
    device,
    frame_width=240,
    shuffle=True,
):
    """This generator takes a spike dataset and generates spiking network input as sparse tensors.

    Args:
        X: The data ( sample x event x 4 ) the last dim holds (timestamp (s), x, y, polarity) tuples
        y: The labels
    """

    labels_ = np.array(y, dtype=int)
    number_of_batches = len(labels_) // batch_size
    sample_index = np.arange(len(labels_))

    # compute discrete firing times
    firing_times = []
    x_pixels = []
    y_pixels = []
    # units_fired = X['units']

    for sample in X:
        firing_times.append(sample["timestamp"])
        x_pixels.append(sample["x"])
        y_pixels.append(sample["y"])

    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter : batch_size * (counter + 1)]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = y_pixels[idx] * frame_width + x_pixels[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse_coo_tensor(
            i, v, torch.Size([batch_size, nb_steps, nb_units])
        ).to(device)
        y_batch = torch.tensor(labels_[batch_index]).to(device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1
