import numpy as np
import torch
import torch.nn as nn

from utils.data_loader import sparse_data_generator_from_hdf5_spikes
from utils.visualization import plot_voltage_traces


class Trainer:
    def __init__(self, model, graph_renderer=None):
        self.model = model
        self.graph_renderer = graph_renderer
        self.device = model.device
        self.batch_size = model.batch_size
        self.nb_steps = model.nb_steps
        self.nb_inputs = model.nb_inputs
        self.max_time = model.max_time

    def train(self, x_data, y_data, nb_epochs=10, lr=1e-3):
        params = [self.model.w1, self.model.w2, self.model.v1]
        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()

        loss_hist = []
        train_accuracy_hist = []

        for e in range(nb_epochs):
            local_loss = []
            local_accuracy = []
            for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
                x_data,
                y_data,
                self.batch_size,
                self.nb_steps,
                self.nb_inputs,
                self.max_time,
                self.device,
            ):
                output, recs = self.model.forward(x_local.to_dense())
                _, spks = recs
                m, _ = torch.max(output, 1)
                log_p_y = log_softmax_fn(m)

                # Calculate the test accuracy
                _, am = torch.max(m, 1)  # argmax over output units
                tmp = np.mean(
                    (y_local == am).detach().cpu().numpy()
                )  # compare to labels
                local_accuracy.append(tmp)

                # Here we set up our regularizer loss
                # The strength paramters here are merely a guess and there should be ample room for improvement by
                # tuning these paramters.
                reg_loss = 2e-6 * torch.sum(spks)  # L1 loss on total number of spikes
                reg_loss += 2e-6 * torch.mean(
                    torch.sum(torch.sum(spks, dim=0), dim=0) ** 2
                )  # L2 loss on spikes per neuron

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local.long()) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())

            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)
            mean_accuracy = np.mean(local_accuracy)
            train_accuracy_hist.append(mean_accuracy)
            # live_plot(loss_hist, title="Loss History", renderer=self.graph_renderer)
            print(
                f"Epoch {e + 1}: Loss = {mean_loss:.4f} \t Training accuracy = {mean_accuracy:.4f}"
            )
        return loss_hist, train_accuracy_hist

    def compute_accuracy(self, x_data, y_data):
        """Computes classification accuracy on supplied data in batches."""
        accs = []
        for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
            x_data,
            y_data,
            self.batch_size,
            self.nb_steps,
            self.nb_inputs,
            self.max_time,
            self.device,
            shuffle=False,
        ):
            output, _ = self.model.forward(x_local.to_dense())
            m, _ = torch.max(output, 1)  # max over time
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
            accs.append(tmp)
        return np.mean(accs)

    def visualize_output(self, x_data, y_data, nb_batches=1):
        batch_counter = 0
        for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
            x_data,
            y_data,
            self.batch_size,
            self.nb_steps,
            self.nb_inputs,
            self.max_time,
            self.device,
            shuffle=False,
        ):
            output, _ = self.model.forward(x_local.to_dense())
            plot_voltage_traces(
                output.detach().cpu().numpy(),
                labels=y_local.detach().cpu().tolist(),
                dim=(1, self.batch_size),
                renderer=self.graph_renderer,
            )

            batch_counter += 1
            if batch_counter == nb_batches:
                break
