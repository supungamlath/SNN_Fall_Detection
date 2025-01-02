import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        optimizer = torch.optim.Adamax(
            self.model.parameters(), lr=lr, betas=(0.9, 0.999)
        )
        scheduler = StepLR(optimizer, step_size=3, gamma=0.90)

        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()

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
                        m, _ = torch.max(output, 1)  # max over time
                        log_p_y = log_softmax_fn(m)
                        _, am = torch.max(m, 1)  # argmax over output units

                        # Convert to numpy arrays
                        y_true = y_local.detach().cpu().numpy()
                        y_pred = am.detach().cpu().numpy()

                        # Aggregate results
                        test_loss.append(loss_fn(log_p_y, y_local.long()).item())
                        test_y_true.extend(y_true)
                        test_y_pred.extend(y_pred)

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
                m, _ = torch.max(output, 1)
                log_p_y = log_softmax_fn(m)
                _, am = torch.max(m, 1)  # argmax over output units

                # Arrays for calculating train metrics
                y_true = y_local.detach().cpu().numpy()
                y_pred = am.detach().cpu().numpy()

                # Aggregate results
                train_y_true.extend(y_true)
                train_y_pred.extend(y_pred)

                # Here we set up our regularizer loss
                # The reg_alpha strength parameter here are merely a guess and there should be ample room for improvement by tuning these paramters.
                reg_loss = 0
                for spks in spk_recs:
                    # L1 loss on total number of spikes
                    reg_loss += reg_alpha * torch.sum(spks)
                    # L2 loss on spikes per neuron
                    reg_loss += reg_alpha * torch.mean(
                        torch.sum(torch.sum(spks, dim=0), dim=0) ** 2
                    )

                # Here we combine supervised loss and the regularizer
                loss_val = loss_fn(log_p_y, y_local.long()) + reg_loss

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

    def compute_metrics(self, all_y_pred, all_y_true):
        """Computes classification accuracy, precision, recall, and F1 score on supplied data in batches."""
        # Convert lists to numpy arrays
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        # Calculate overall metrics
        accuracy = accuracy_score(all_y_true, all_y_pred)
        precision = precision_score(
            all_y_true, all_y_pred, average="binary", zero_division=0
        )
        recall = recall_score(all_y_true, all_y_pred, average="binary", zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, average="binary", zero_division=0)

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
