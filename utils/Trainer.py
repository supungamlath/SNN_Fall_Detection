import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from utils.helpers import EarlyStopping

# Deterministic behavior for reproducibility
torch.manual_seed(0)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.is_done = False  # Flag to stop training early
        self.early_stopper = EarlyStopping(patience=10, min_delta=0.0001)
        self.nb_steps = model.nb_steps
        self.chunk_size = self.nb_steps // 60  # Number of timesteps is split into 60 chunks, 1 chunk per second

    def train(
        self,
        train_dataloader,
        nb_epochs=10,
        lr=1e-3,
        use_regularizer=True,
        regularizer_alpha=2e-6,
        step_lr_size=3,
        step_lr_gamma=0.90,
        stop_early=False,
        dataset_bias_ratio=46.0,
        evaluate_dataloader=None,
        evaluate_callback=None,
        train_callback=None,
    ):
        optimizer = torch.optim.NAdam(self.model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=step_lr_size, gamma=step_lr_gamma)

        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, dataset_bias_ratio]).to(self.model.device, self.model.dtype)
        )

        train_metrics = Metrics()
        dev_metrics = Metrics()

        for epoch in range(nb_epochs):
            if self.is_done:
                break

            print(f"Epoch: {epoch + 1}")
            epoch_start_time = time.time()  # Start timing the epoch

            # Evaluate loop
            if evaluate_dataloader is not None:
                with torch.inference_mode():
                    for x_local, y_local in evaluate_dataloader:

                        x_local = x_local.to(self.model.device, self.model.dtype)
                        y_local = y_local.to(self.model.device, self.model.dtype)

                        output, _ = self.model.forward(x_local.to_dense())
                        output = output[:, : self.nb_steps, :].reshape(-1, 60, self.chunk_size, 2).mean(dim=2)

                        # Get the max value for each second as the prediction
                        y_pred = torch.argmax(output, dim=2)

                        # Cross Entropy Loss function expects the input to be of shape (N, C, L)
                        ce_loss = loss_fn(output.permute(0, 2, 1), y_local.long())

                        dev_metrics.update(
                            y_pred.cpu().detach().numpy(), y_local.cpu().detach().numpy(), ce_loss.item()
                        )

                    dev_metrics_dict = dev_metrics.compute()
                    print(f"Dev Set Metrics : {dev_metrics_dict}")

                    evaluate_callback and evaluate_callback(dev_metrics_dict, epoch)
                    dev_metrics.reset()

                if stop_early and self.early_stopper(dev_metrics_dict["loss"]):
                    self.is_done = True
                    print(self.early_stopper.status)

            # Training loop
            for x_local, y_local in train_dataloader:
                optimizer.zero_grad()  # Clears gradients to prevent accumulation of gradients from multiple backward passes.
                x_local = x_local.to(self.model.device, self.model.dtype)
                y_local = y_local.to(self.model.device, self.model.dtype)

                output, recs = self.model.forward(x_local.to_dense())
                spk_recs, _ = recs
                output = output[:, : self.nb_steps, :].reshape(-1, 60, self.chunk_size, 2).mean(dim=2)

                # Get the max value for each second as the prediction
                y_pred = torch.argmax(output, dim=2)

                # Calculate regularizer loss
                # The reg_alpha parameter controls the strength of the regularizer
                reg_loss = 0
                if use_regularizer:
                    for spks in spk_recs:
                        # L1 loss on total number of spikes
                        reg_loss += regularizer_alpha * torch.sum(spks)
                        # L2 loss on spikes per neuron
                        reg_loss += regularizer_alpha * torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0) ** 2)

                # Combine cross entropy loss and regularizer loss
                total_loss = loss_fn(output.permute(0, 2, 1), y_local.long()) + reg_loss

                train_metrics.update(y_pred.cpu().detach().numpy(), y_local.cpu().detach().numpy(), total_loss.item())

                total_loss.backward()  # Computes gradients and stores them in the .grad attributes of the parameters.
                optimizer.step()  # Updates the parameters using the gradients stored in .grad.

            scheduler.step()

            train_metrics_dict = train_metrics.compute()
            print(f"Train Set Metrics : {train_metrics_dict}")

            train_callback and train_callback(train_metrics_dict, epoch)
            train_metrics.reset()

            # End timing the epoch and print the duration
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_minutes = int(epoch_duration // 60)
            epoch_seconds = int(epoch_duration % 60)
            print(f"Epoch {epoch + 1} took {epoch_minutes} minutes {epoch_seconds} seconds")

    def test(self, test_dataloader):
        test_metrics = Metrics()
        with torch.inference_mode():
            for x_local, y_local in test_dataloader:

                x_local = x_local.to(self.model.device, self.model.dtype)
                y_local = y_local.to(self.model.device, self.model.dtype)

                output, _ = self.model.forward(x_local.to_dense())
                output = output[:, : self.chunk_size * 60, :].reshape(-1, 60, self.chunk_size, 2).mean(dim=2)

                # Get the max value for each second as the prediction
                y_pred = torch.argmax(output, dim=2)

                # Set the loss to 0 as we are not calculating it here
                test_metrics.update(y_pred.cpu().detach().numpy(), y_local.cpu().detach().numpy(), 0)

            test_metrics_dict = test_metrics.compute()
            print(f"Test Set Metrics : {test_metrics_dict}")
            test_metrics.reset()
        return test_metrics_dict


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
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

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
