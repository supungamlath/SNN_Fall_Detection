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
        self.frame_height = dataset.frame_height
        self.batch_size = kwargs.get("batch_size")
        print(f"Initializing DataLoader of size {len(dataset)}")

        super().__init__(
            dataset,
            *args,
            collate_fn=self.sparse_data_generator,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            **kwargs,
        )

    def sparse_data_generator(
        self,
        batch,
    ):
        time_bins = np.linspace(0, self.time_duration, num=self.nb_steps)

        coo = [[] for _ in range(4)]
        labels = []
        for i, datapoint in enumerate(batch):
            sample, label = datapoint
            times = np.digitize(sample["t"], time_bins)
            pixels_x = sample["x"]
            pixels_y = sample["y"]
            batch = [i for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(pixels_x)
            coo[3].extend(pixels_y)
            labels.append(label)

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse_coo_tensor(
            i, v, torch.Size([self.batch_size, self.nb_steps, self.frame_width, self.frame_height])
        )
        y_batch = torch.tensor(labels)

        return X_batch, y_batch
