import torch
import torch.nn as nn
import snntorch as snn
import numpy as np


class SNNTorchConv(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        nb_steps: int,
        time_step=1e-2,
        tau_mem=10e-2,
        tau_syn=5e-2,
    ):
        super(SNNTorchConv, self).__init__()

        self.nb_steps = nb_steps
        self.alpha = float(np.exp(-time_step / tau_syn))
        self.beta = float(np.exp(-time_step / tau_mem))
        print(f"LIF Parameter alpha: {self.alpha}")
        print(f"LIF Parameter beta: {self.beta}")

        # Initialize layers
        # Input shape: [batch_size, 1, 240, 180]
        self.conv1 = nn.Conv2d(1, 4, 5, padding="same")  # Output shape: [batch_size, 4, 240, 180]
        self.mp1 = nn.MaxPool2d(2)  # Output shape: [batch_size, 4, 120, 90]
        self.lif1 = snn.Synaptic(alpha=self.alpha, beta=self.beta)
        self.conv2 = nn.Conv2d(4, 4, 5, padding="same")  # Output shape: [batch_size, 4, 120, 90]
        self.mp2 = nn.MaxPool2d(2)  # Output shape: [batch_size, 4, 60, 45]
        self.lif2 = snn.Synaptic(alpha=self.alpha, beta=self.beta)
        self.fc = nn.Linear(4 * 60 * 45, num_outputs)
        self.lif3 = snn.Synaptic(alpha=self.alpha, beta=self.beta)

        # Move the model to the GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.to(self.device, self.dtype)

    def forward(self, x):

        # Initialize hidden states at t=0
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()

        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        # Add channel dimension to input: [batch_size, timesteps, width, height] -> [batch_size, timesteps, 1, width, height]
        x = x.unsqueeze(2)

        for step in range(self.nb_steps):
            cur1 = self.conv1(x[:, step])
            spk1, syn1, mem1 = self.lif1(self.mp1(cur1), syn1, mem1)
            cur2 = self.conv2(spk1)
            spk2, syn2, mem2 = self.lif2(self.mp2(cur2), syn2, mem2)
            cur3 = self.fc(spk2.flatten(1))
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(mem3_rec, dim=1), torch.stack(spk3_rec, dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
