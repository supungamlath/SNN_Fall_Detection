import numpy as np
import torch
from torch.utils.data import DataLoader


# Custom DataLoader that uses a sparse_data_generator as the collate function
class SpikingDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        dataset = args[0]
        self.nb_units = dataset.nb_pixels
        self.time_duration = dataset.time_duration
        self.nb_steps = kwargs["nb_steps"]
        self.frame_width = dataset.frame_width
        self.batch_size = kwargs["batch_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing DataLoader using device:", self.device)

        kwargs["collate_fn"] = self.sparse_data_generator
        super().__init__(*args, **kwargs)

    def sparse_data_generator(
        self,
        batch,
    ):
        time_bins = np.linspace(0, self.time_duration, num=self.nb_steps)

        coo = [[] for _ in range(3)]
        labels = []
        for i, datapoint in enumerate(batch):
            sample, label = datapoint
            times = np.digitize(sample["timestamp"], time_bins)
            units = sample["y"] * self.frame_width + sample["x"]
            batch = [i for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)
            labels.append(label)

        i = torch.LongTensor(coo).to(self.device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)

        X_batch = torch.sparse_coo_tensor(i, v, torch.Size([self.batch_size, self.nb_steps, self.nb_units])).to(
            self.device
        )
        y_batch = torch.tensor(labels).to(self.device)

        return X_batch.to(device=self.device), y_batch.to(device=self.device)
