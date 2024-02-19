import numpy as np
import torch
import torch.nn as nn

from utils.visualization import plot_voltage_traces


class Trainer:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.batch_size = model.batch_size
        self.nb_steps = model.nb_steps
        self.nb_inputs = model.nb_inputs
        self.max_time = model.max_time

    def train(
        self,
        dataloader,
        nb_epochs=10,
        lr=1e-3,
        evaluate_loader=None,
        callback_fn=None,
    ):
        params = [self.model.w1, self.model.w2, self.model.v1]
        optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9, 0.999))

        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()

        loss_hist = []
        train_accuracy_hist = []
        test_accuracy_hist = []

        for e in range(nb_epochs):
            local_loss = []
            local_accuracy = []
            print(f"Epoch: {e + 1}")
            if evaluate_loader is not None:
                test_accuracy = self.compute_accuracy(evaluate_loader)
                test_accuracy_hist.append(test_accuracy)
                print(f"Test accuracy = {test_accuracy:.4f}")
            for x_local, y_local in dataloader:
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
            print(f"Train accuracy = {mean_accuracy:.4f}")
            print(f"Loss = {mean_loss:.4f}\n")
            if evaluate_loader is not None:
                callback_fn(loss_hist, train_accuracy_hist, test_accuracy_hist)
            else:
                callback_fn(loss_hist, train_accuracy_hist)

        return loss_hist, train_accuracy_hist, test_accuracy_hist

    def compute_accuracy(self, dataloader):
        """Computes classification accuracy on supplied data in batches."""
        accs = []
        for x_local, y_local in dataloader:
            output, _ = self.model.forward(x_local.to_dense())
            m, _ = torch.max(output, 1)  # max over time
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
            accs.append(tmp)
        return np.mean(accs)

    def visualize_output(self, dataloader, nb_batches=1):
        batch_counter = 0
        for x_local, y_local in dataloader:
            output, _ = self.model.forward(x_local.to_dense())
            two_maxims, _ = torch.max(output, 1)  # max over time
            _, model_preds = torch.max(two_maxims, 1)  # argmax over output units
            diff = torch.abs(two_maxims[:, 0] - two_maxims[:, 1])
            plot_voltage_traces(
                mem=output.detach().cpu().numpy(),
                diff=diff.detach().cpu().numpy(),
                labels=model_preds.detach().cpu().tolist(),
                dim=(1, self.batch_size),
            )

            batch_counter += 1
            if batch_counter == nb_batches:
                break
