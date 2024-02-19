import torch
import torch.nn as nn
import numpy as np


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


class SNN(nn.Module):
    def __init__(
        self,
        nb_inputs,
        nb_hidden,
        nb_outputs,
        batch_size,
        max_time,
        nb_steps,
        dtype=torch.float,
        time_step=1e-2,
        tau_mem=10e-2,
        tau_syn=5e-2,
    ):
        super(SNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing SNN Model using device:", self.device)
        self.dtype = dtype

        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.batch_size = batch_size
        self.max_time = max_time
        self.nb_steps = nb_steps

        self.time_step = time_step
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn

        self.alpha = float(np.exp(-time_step / tau_syn))
        self.beta = float(np.exp(-time_step / tau_mem))

        # Initialize weights
        self.init_weights()

        self.spike_fn = SurrGradSpike.apply

    def init_weights(self):
        weight_scale = 0.2

        self.w1 = nn.Parameter(
            torch.empty(
                (self.nb_inputs, self.nb_hidden), device=self.device, dtype=self.dtype
            )
        )
        nn.init.normal_(self.w1, mean=0.0, std=weight_scale / np.sqrt(self.nb_inputs))

        self.w2 = nn.Parameter(
            torch.empty(
                (self.nb_hidden, self.nb_outputs), device=self.device, dtype=self.dtype
            )
        )
        nn.init.normal_(self.w2, mean=0.0, std=weight_scale / np.sqrt(self.nb_hidden))

        self.v1 = nn.Parameter(
            torch.empty(
                (self.nb_hidden, self.nb_hidden), device=self.device, dtype=self.dtype
            )
        )
        nn.init.normal_(self.v1, mean=0.0, std=weight_scale / np.sqrt(self.nb_hidden))

    def forward(self, inputs):
        syn = torch.zeros(
            (self.batch_size, self.nb_hidden), device=self.device, dtype=self.dtype
        )
        mem = torch.zeros(
            (self.batch_size, self.nb_hidden), device=self.device, dtype=self.dtype
        )

        mem_rec = []
        spk_rec = []

        # Compute hidden layer activity
        out = torch.zeros(
            (self.batch_size, self.nb_hidden), device=self.device, dtype=self.dtype
        )
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w1))
        for t in range(self.nb_steps):
            h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, self.v1))
            mthr = mem - 1.0
            out = self.spike_fn(mthr)
            rst = out.detach()  # We do not want to backprop through the reset

            new_syn = self.alpha * syn + h1
            new_mem = (self.beta * mem + syn) * (1.0 - rst)

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        # Readout layer
        h2 = torch.einsum("abc,cd->abd", (spk_rec, self.w2))
        flt = torch.zeros(
            (self.batch_size, self.nb_outputs), device=self.device, dtype=self.dtype
        )
        out = torch.zeros(
            (self.batch_size, self.nb_outputs), device=self.device, dtype=self.dtype
        )
        out_rec = [out]
        for t in range(self.nb_steps):
            new_flt = self.alpha * flt + h2[:, t]
            new_out = self.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        other_recs = [mem_rec, spk_rec]
        return out_rec, other_recs

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
