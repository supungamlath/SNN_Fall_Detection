import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from utils.helpers import EarlyStopping
from utils.visualization import plot_voltage_traces


class Trainer:
    def __init__(self, model):
        self.model = model
        self.is_done = False  # Flag to stop training early
        self.early_stopper = EarlyStopping()

    def train(
        self,
        train_dataloader,
        nb_epochs=10,
        lr=1e-3,
        reg_alpha=2e-6,
        evaluate_dataloader=None,
        callback_fn=None,
        stop_early=False,
    ):
        optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = StepLR(optimizer, step_size=3, gamma=0.90)

        loss_fn = nn.CrossEntropyLoss()

        test_metrics = Metrics()
        train_metrics = Metrics()
        train_metrics_hist = []
        test_metrics_hist = []

        chunk_size = 3000 // 60

        for e in range(nb_epochs):
            if self.is_done:
                break

            print(f"Epoch: {e + 1}")

            if evaluate_dataloader is not None:
                for x_local, y_local in evaluate_dataloader:
                    with torch.no_grad():
                        output, _ = self.model.forward(x_local.to_dense())
                        output = output[:, : chunk_size * 60, :].reshape(7, 60, chunk_size, 2).mean(dim=2)

                        # Get the max value for each second as the prediction
                        y_pred = torch.argmax(output, dim=2)

                        # Cross Entropy Loss function expects the input to be of shape (N, C, L)
                        ce_loss = loss_fn(output.permute(0, 2, 1), y_local.long())

                        test_metrics.update(
                            y_pred.cpu().detach().numpy(), y_local.cpu().detach().numpy(), ce_loss.item()
                        )

                test_metrics_dict = test_metrics.compute()
                test_metrics_hist.append(test_metrics_dict)
                print(f"Test metrics : {test_metrics_dict}")
                test_metrics.reset()

                if stop_early and self.early_stopper(test_metrics_dict["loss"]):
                    self.is_done = True
                    print(self.early_stopper.status)

            for x_local, y_local in train_dataloader:
                output, recs = self.model.forward(x_local.to_dense())
                spk_recs, _ = recs
                output = output[:, : chunk_size * 60, :].reshape(7, 60, chunk_size, 2).mean(dim=2)

                # Get the max value for each second as the prediction
                y_pred = torch.argmax(output, dim=2)

                # Here we set up our regularizer loss
                # The reg_alpha strength parameter here are merely a guess and there should be ample room for improvement by tuning these parameters.
                reg_loss = 0
                for spks in spk_recs:
                    # L1 loss on total number of spikes
                    reg_loss += reg_alpha * torch.sum(spks)
                    # L2 loss on spikes per neuron
                    reg_loss += reg_alpha * torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0) ** 2)

                # Combine CE loss and the regularizer
                total_loss = loss_fn(output.permute(0, 2, 1), y_local.long()) + reg_loss

                train_metrics.update(y_pred.cpu().detach().numpy(), y_local.cpu().detach().numpy(), total_loss.item())

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            scheduler.step()

            train_metrics_dict = train_metrics.compute()
            train_metrics_hist.append(train_metrics_dict)
            print(f"Train metrics : {train_metrics_dict}")

            train_metrics.reset()

            if callback_fn is not None:
                if evaluate_dataloader is not None:
                    callback_fn(train_metrics_hist, test_metrics_hist)
                else:
                    callback_fn(train_metrics_hist)

        return train_metrics_hist, test_metrics_hist

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
                dim=(1, x_local.size(0)),
            )

            batch_counter += 1
            if batch_counter == nb_batches:
                break


class Metrics:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.loss = []

    def update(self, batch_y_pred, batch_y_true, batch_loss):
        """Update the state with predictions and true labels."""
        self.y_pred.extend(batch_y_pred)
        self.y_true.extend(batch_y_true)
        self.loss.append(batch_loss)

    def compute(self):
        """Compute metrics based on the current state."""
        y_pred = np.array(self.y_pred).flatten()
        y_true = np.array(self.y_true).flatten()

        loss = np.mean(self.loss)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="binary")
        recall = recall_score(y_true, y_pred, average="binary")
        f1 = f1_score(y_true, y_pred, average="binary")

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def reset(self):
        """Reset the internal state."""
        self.y_true = []
        self.y_pred = []
        self.loss = []
