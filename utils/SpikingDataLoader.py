import numpy as np
import torch
from torch.utils.data import DataLoader


# Custom DataLoader that uses a sparse_data_generator as the collate function
class SpikingDataLoader(DataLoader):
    def __init__(self, dataset, nb_steps, *args, **kwargs):
        self.nb_units = dataset.nb_pixels
        self.time_duration = dataset.time_duration
        self.nb_steps = nb_steps
        self.frame_width = dataset.frame_width
        self.batch_size = kwargs.get("batch_size")
        print(f"Initializing DataLoader of size {len(dataset)}")

        kwargs["collate_fn"] = self.sparse_data_generator
        kwargs["num_workers"] = 4
        kwargs["pin_memory"] = True
        super().__init__(dataset, *args, **kwargs)

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

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse_coo_tensor(i, v, torch.Size([self.batch_size, self.nb_steps, self.nb_units]))
        y_batch = torch.tensor(labels)

        return X_batch, y_batch
