import torch
import torch.nn as nn
import snntorch as snn
import numpy as np


class SNNTorchConv(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        nb_steps: int,
        time_step=1e-2,
        tau_mem=10e-2,
    ):
        super(SNNTorchConv, self).__init__()

        self.nb_steps = nb_steps
        self.beta = float(np.exp(-time_step / tau_mem))
        print(f"LIF Parameter beta: {self.beta}")

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 8, 5, padding="same")
        self.lif1 = snn.Leaky(beta=self.beta)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 24, 5, padding="same")
        self.lif2 = snn.Leaky(beta=self.beta)
        self.mp2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(7 * 7 * 24, 10)
        self.lif3 = snn.Leaky(beta=self.beta)

        # Move the model to the GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.to(self.device, self.dtype)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        for step in range(self.nb_steps):
            cur1 = self.conv1(x[:, step])
            spk1, mem1 = self.lif1(self.mp1(cur1), mem1)
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(self.mp2(cur2), mem2)
            cur3 = self.fc(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(mem3_rec, dim=0), torch.stack(spk3_rec, dim=0)
