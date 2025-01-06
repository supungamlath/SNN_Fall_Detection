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


class SpikingHiddenLayer(nn.Module):
    def __init__(self, nb_inputs, nb_hidden, nb_steps, alpha, beta):
        super(SpikingHiddenLayer, self).__init__()
        self.nb_inputs = nb_inputs
        self.nb_hidden = nb_hidden
        self.nb_steps = nb_steps
        self.alpha = alpha
        self.beta = beta

        self.spike_fn = SurrGradSpike.apply

        weight_scale = 0.2

        self.w_input = nn.Parameter(torch.empty((nb_inputs, nb_hidden)))
        nn.init.normal_(self.w_input, mean=0.0, std=weight_scale / np.sqrt(nb_inputs))
        self.w_hidden = nn.Parameter(torch.empty((nb_hidden, nb_hidden)))
        nn.init.normal_(self.w_hidden, mean=0.0, std=weight_scale / np.sqrt(nb_hidden))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        device = self.w_input.device
        dtype = self.w_input.dtype

        syn = torch.zeros((batch_size, self.nb_hidden), device=device, dtype=dtype)
        mem = torch.zeros((batch_size, self.nb_hidden), device=device, dtype=dtype)
        out = torch.zeros((batch_size, self.nb_hidden), device=device, dtype=dtype)

        mem_rec = []
        spk_rec = []

        h1_from_input = torch.einsum("abc,cd->abd", (inputs, self.w_input))
        for t in range(self.nb_steps):
            h1 = h1_from_input[:, t] + torch.einsum("ab,bc->ac", (out, self.w_hidden))
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
        return spk_rec, mem_rec


class SpikingReadoutLayer(nn.Module):
    def __init__(self, nb_hidden, nb_outputs, nb_steps, alpha, beta):
        super(SpikingReadoutLayer, self).__init__()
        self.nb_hidden = nb_hidden
        self.nb_outputs = nb_outputs
        self.nb_steps = nb_steps
        self.alpha = alpha
        self.beta = beta

        weight_scale = 0.2

        self.output_weights = nn.Parameter(torch.empty((self.nb_hidden, self.nb_outputs)))
        nn.init.normal_(self.output_weights, mean=0.0, std=weight_scale / np.sqrt(self.nb_hidden))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        device = self.output_weights.device
        dtype = self.output_weights.dtype

        flt = torch.zeros((batch_size, self.nb_outputs), device=device, dtype=dtype)
        out = torch.zeros((batch_size, self.nb_outputs), device=device, dtype=dtype)

        out_rec = [out]

        h2 = torch.einsum("abc,cd->abd", (inputs, self.output_weights))
        for t in range(self.nb_steps):
            new_flt = self.alpha * flt + h2[:, t]
            new_out = self.beta * out + flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        return out_rec


class SpikingNN(nn.Module):
    def __init__(
        self,
        layer_sizes: list,
        nb_steps: int,
        time_step=1e-2,
        tau_mem=10e-2,
        tau_syn=5e-2,
    ):
        super(SpikingNN, self).__init__()

        self.nb_steps = nb_steps
        self.alpha = float(np.exp(-time_step / tau_syn))
        self.beta = float(np.exp(-time_step / tau_mem))

        # Using ModuleList for hidden layers
        nb_hidden_layers = len(layer_sizes) - 2
        hidden_layers = []
        for i in range(nb_hidden_layers):
            hidden_layers.append(
                SpikingHiddenLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    nb_steps,
                    self.alpha,
                    self.beta,
                )
            )
        self.hidden_layers = nn.ModuleList(hidden_layers)
        # Readout layer
        self.readout_layer = SpikingReadoutLayer(layer_sizes[-2], layer_sizes[-1], nb_steps, self.alpha, self.beta)

        # Move the model to the GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.to(self.device, self.dtype)

    def forward(self, x):
        # Forward pass through hidden layers
        spk_recs = []
        mem_recs = []
        for hidden_layer in self.hidden_layers:
            x, mem_rec = hidden_layer(x)
            spk_recs.append(x)
            mem_recs.append(mem_rec)
        # Forward pass through the readout layer
        out = self.readout_layer(x)
        return out, (spk_recs, mem_recs)

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path, weights_only=True)
