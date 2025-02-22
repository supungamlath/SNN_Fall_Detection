import torch
import torch.nn as nn
import snntorch as snn
import numpy as np


class SNNTorchSyn(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        nb_steps: int,
        time_step=1e-2,
        tau_mem=10e-2,
        tau_syn=5e-2,
    ):
        super(SNNTorchSyn, self).__init__()

        self.nb_steps = nb_steps
        self.alpha = float(np.exp(-time_step / tau_syn))
        self.beta = float(np.exp(-time_step / tau_mem))
        print(f"LIF Parameter alpha: {self.alpha}")
        print(f"LIF Parameter beta: {self.beta}")

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Synaptic(alpha=self.alpha, beta=self.beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Synaptic(alpha=self.alpha, beta=self.beta)

        # Move the model to the GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.to(self.device, self.dtype)

    def forward(self, x):

        # Initialize hidden states at t=0
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        # Flatten the last two dimensions into one (height * width)
        x = torch.flatten(x, start_dim=-2)

        for step in range(self.nb_steps):
            cur1 = self.fc1(x[:, step])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(mem2_rec, dim=1), torch.stack(spk2_rec, dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
