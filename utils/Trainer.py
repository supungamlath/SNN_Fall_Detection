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
        self.early_stopper = EarlyStopping()
        self.is_done = False

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

        loss_hist = []
        train_metrics_hist = []
        test_metrics_hist = []

        for e in range(nb_epochs):
            local_loss = []
            train_y_true = []
            train_y_pred = []

            if self.is_done:
                break

            print(f"Epoch: {e + 1}")

            if evaluate_dataloader is not None:
                test_y_true = []
                test_y_pred = []
                test_loss = []
                for x_local, y_local in evaluate_dataloader:
                    with torch.no_grad():
                        output, _ = self.model.forward(x_local.to_dense())
                        chunk_size = 3000 // 60
                        output = output.reshape(7, 60, chunk_size, 2).mean(dim=2)

                        # Arrays for calculating test metrics
                        y_true = y_local.long()
                        y_pred = torch.sigmoid(output)

                        # Aggregate results
                        bce_loss = loss_fn(output.permute(0, 2, 1), y_true)
                        test_loss.append(bce_loss.item())
                        test_y_true.extend(y_true.cpu().detach().numpy())
                        test_y_pred.extend(y_pred.cpu().detach().numpy())

                mean_test_loss = np.mean(test_loss)
                test_metrics = self.compute_metrics(test_y_pred, test_y_true)
                test_metrics["loss"] = mean_test_loss
                test_metrics_hist.append(test_metrics)
                print(f"Test metrics : {test_metrics}")

                if stop_early and self.early_stopper(mean_test_loss):
                    self.is_done = True
                    print(self.early_stopper.status)

            for x_local, y_local in train_dataloader:
                output, recs = self.model.forward(x_local.to_dense())
                spk_recs, _ = recs
                chunk_size = 3000 // 60
                output = output.reshape(7, 60, chunk_size, 2).mean(dim=2)

                # Arrays for calculating train metrics
                y_true = y_local.long()
                y_pred = torch.sigmoid(output)

                # Aggregate results
                train_y_true.extend(y_true.cpu().detach().numpy())
                train_y_pred.extend(y_pred.cpu().detach().numpy())

                # Here we set up our regularizer loss
                # The reg_alpha strength parameter here are merely a guess and there should be ample room for improvement by tuning these paramters.
                reg_loss = 0
                for spks in spk_recs:
                    # L1 loss on total number of spikes
                    reg_loss += reg_alpha * torch.sum(spks)
                    # L2 loss on spikes per neuron
                    reg_loss += reg_alpha * torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0) ** 2)

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(output.permute(0, 2, 1), y_true) + reg_loss

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                local_loss.append(loss_val.item())

            scheduler.step()
            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)

            train_metrics = self.compute_metrics(train_y_pred, train_y_true)
            train_metrics["loss"] = mean_loss
            train_metrics_hist.append(train_metrics)

            print(f"Train metrics : {train_metrics}")

            if callback_fn is not None:
                if evaluate_dataloader is not None:
                    callback_fn(train_metrics_hist, test_metrics_hist)
                else:
                    callback_fn(train_metrics_hist)

        return train_metrics_hist, test_metrics_hist

    def compute_metrics(self, y_pred, y_true):
        # Flatten predictions and true labels for metric calculation
        y_pred = np.argmax(np.array(y_pred), axis=-1).flatten()
        y_true = np.array(y_true).flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="binary")
        recall = recall_score(y_true, y_pred, average="binary")
        f1 = f1_score(y_true, y_pred, average="binary")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

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
