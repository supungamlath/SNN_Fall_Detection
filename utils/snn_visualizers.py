from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.metrics import confusion_matrix

config = {
    "toImageButtonOptions": {
        "format": "svg",  # one of png, svg, jpeg, webp
    }
}


def visualize_events(
    data: np.ndarray,
    labels: np.ndarray,
    time_seconds: int = None,
):
    """
    Visualizes a 3D vector (timesteps, height, width) using Plotly with an option
    to reduce the timesteps and display averaged frames. Ensures all plots share the same intensity scale.

    Parameters:
        data (np.ndarray): Input data of shape (timesteps, height, width).
        labels (np.ndarray): Labels for each timestep.
        time_seconds (int): Number of timesteps to reduce to. If None, all timesteps are used.
    """
    timesteps, width, height = data.shape

    # Reshape and sum events to reduce timesteps
    assert timesteps % time_seconds == 0, "Timesteps must be divisible by time_seconds."
    group_size = timesteps // time_seconds
    data = data.reshape(time_seconds, group_size, width, height).sum(axis=1)

    timesteps = data.shape[0]  # Update timesteps after reduction

    # Validate labels
    assert len(labels) == timesteps, "Number of labels must match the number of timesteps."

    # Determine global min and max values for the intensity scale
    zmin = data.min()
    zmax = data.max()

    # Create a subplot grid
    fig = sp.make_subplots(
        rows=1,
        cols=timesteps,
        subplot_titles=[f"Label: {int(labels[i])}" for i in range(timesteps)],
        horizontal_spacing=0.0005,
    )

    # Add frames to the subplot
    for i in range(timesteps):
        frame = np.rot90(data[i])  # Plotly expects the data to be rotated

        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorscale="gray",
                showscale=(i == 0),  # Show scale only on one plot
                showlegend=False,
                colorbar=dict(len=1.5, title="Intensity"),
                zmin=zmin,
                zmax=zmax,
            ),
            row=1,
            col=i + 1,
        )

        # Set axis constraints for each subplot
        fig.update_yaxes(
            scaleanchor=f"x{i + 1}",  # Tie y-axis scale to x-axis scale
            row=1,
            col=i + 1,
        )

    # Update layout
    fig.update_layout(
        title="Ground Truth",
        height=250,
        width=timesteps * 100,
    )
    fig.update_annotations(font_size=14)
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False)

    fig.show(config=config)


def visualize_events_grid(data, time_seconds=60, columns=10):
    timesteps, height, width = data.shape
    # Reshape and average data to reduce timesteps
    if time_seconds is not None:
        assert timesteps % time_seconds == 0, "Timesteps must be divisible by time_seconds."
        group_size = timesteps // time_seconds
        data = data.reshape(time_seconds, group_size, height, width).mean(axis=1)

    timesteps = data.shape[0]  # Update timesteps after reduction

    # Define zmin and zmax for legend
    zmin = np.min(data)
    zmax = np.max(data)

    # Create a subplot grid
    rows = int(np.ceil(timesteps / columns))
    fig = sp.make_subplots(
        rows=rows, cols=columns, subplot_titles=[f"Timestep {i*group_size}" for i in range(timesteps)]
    )

    for i in range(timesteps):
        row = i // columns + 1
        col = i % columns + 1
        fig.add_trace(
            go.Heatmap(
                z=np.flipud(data[i]),
                colorscale="Viridis",
                colorbar=dict(len=1.0, title="Intensity"),
                zmin=zmin,
                zmax=zmax,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Heatmap of Images at Every 25 Timesteps",
        height=600,
        width=1000,
        font=dict(size=10),
    )
    fig.update_annotations(font_size=10)
    fig.update_xaxes(visible=False)  # Hide x-axes
    fig.update_yaxes(visible=False)  # Hide y-axes

    fig.show(config=config)


def plot_snn_activity(mem_rec, spk_rec, timestep_axis_range=[0, 1500], time_axis_range=[0, 60], time_seconds=60):
    """
    Visualizes the membrane potentials and spikes over time.

    Parameters:
        mem_rec (numpy.ndarray): Membrane potentials of shape (T, N).
        spk_rec (numpy.ndarray): Spikes of shape (T, N).
    """
    # Ensure the inputs are of the expected shape
    assert mem_rec.shape == spk_rec.shape, "mem_rec and spk_rec must have the same shape"

    timesteps, num_neurons = mem_rec.shape

    time = list(range(timesteps))

    # Membrane Potentials Figure
    mem_fig = go.Figure()
    for neuron_idx in range(num_neurons):
        mem_fig.add_trace(go.Scatter(x=time, y=mem_rec[:, neuron_idx], mode="lines", name=f"Neuron {neuron_idx}"))

    mem_fig.update_layout(
        title="Membrane Potentials Over Timesteps",
        xaxis_title="Time (Timesteps)",
        yaxis_title="Membrane Potential",
        legend=dict(orientation="h", x=0, y=-0.2),
        height=500,
        width=1000,
    )
    mem_fig.update_xaxes(range=timestep_axis_range, dtick=timesteps // time_seconds)

    # Spikes Figure
    spk_fig = go.Figure()
    for neuron_idx in range(num_neurons):
        spike_times = np.where(spk_rec[:, neuron_idx] == 1.0)[0]
        spk_fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=[neuron_idx] * len(spike_times),
                mode="markers",
                name=f"Neuron {neuron_idx}",
                marker=dict(size=6),
            )
        )

    spk_fig.update_layout(
        title="Spikes Over Timesteps",
        xaxis_title="Time (Timesteps)",
        yaxis_title="Neuron Index",
        legend=dict(orientation="h", x=0, y=-0.2),
        height=500,
        width=1000,
    )
    spk_fig.update_xaxes(range=timestep_axis_range, dtick=timesteps // time_seconds)

    spk_fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(0, num_neurons)),
        ticktext=[f"Neuron {i}" for i in range(num_neurons)],
    )

    # Time (Seconds) vs Spike Counts Figure
    timesteps_per_s = timesteps // time_seconds
    spk_rec = spk_rec.reshape(time_seconds, timesteps_per_s, num_neurons).sum(axis=1)

    spk_counts_fig = go.Figure()
    for neuron_idx in range(num_neurons):
        spk_counts_fig.add_trace(
            go.Bar(
                x=[i + 0.5 for i in range(0, time_seconds)],  # Shift each bar by 0.5,
                y=spk_rec[:, neuron_idx],
                name=f"Neuron {neuron_idx}",
            )
        )

    spk_counts_fig.update_layout(
        title="Spike Counts per Neuron Over Time",
        xaxis_title="Time (Seconds)",
        yaxis_title="Spike Count",
        legend=dict(orientation="h", x=0, y=-0.2),
        barmode="stack",
        height=500,
        width=1000,
    )
    spk_counts_fig.update_xaxes(range=time_axis_range, dtick=1)

    # Display figures
    mem_fig.show(config=config)
    spk_fig.show(config=config)
    spk_counts_fig.show(config=config)


def plot_snn_activity_combined(
    mem_rec, spk_rec, timestep_axis_range=[0, 1501], time_axis_range=[0, 61], time_seconds=60
):
    """
    Visualizes the membrane potentials with spikes marked on the same figure and spike counts over time.
    """

    # Ensure the inputs are of the expected shape
    assert mem_rec.shape == spk_rec.shape, "mem_rec and spk_rec must have the same shape"

    timesteps, num_neurons = mem_rec.shape

    time = list(range(timesteps))

    # Define a gradient between blue and red for neuron colors
    colors = [
        f"rgb({int(255 * idx / (num_neurons - 1))}, 0, {int(255 * (1 - idx / (num_neurons - 1)))})"
        for idx in range(num_neurons)
    ]

    # Combined Membrane Potentials and Spikes Figure
    mem_spk_fig = go.Figure()
    for neuron_idx in range(num_neurons):
        # Add membrane potential curve
        mem_spk_fig.add_trace(
            go.Scatter(
                x=time,
                y=mem_rec[:, neuron_idx],
                mode="lines",
                name=f"Neuron {neuron_idx} Potential",
                line=dict(color=colors[neuron_idx]),
            )
        )

        # Add spike events as markers
        spike_times = np.where(spk_rec[:, neuron_idx] == 1.0)[0]
        mem_spk_fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=mem_rec[spike_times, neuron_idx],
                mode="markers",
                name=f"Neuron {neuron_idx} Spikes",
                marker=dict(symbol="x", size=6, color=colors[neuron_idx]),
            )
        )

    mem_spk_fig.update_layout(
        title="Membrane Potentials with Spike Events",
        xaxis=dict(title="Time (Timesteps)", side="top"),
        yaxis_title="Membrane Potential",
        legend=dict(orientation="h", x=0, y=-0.2),
        height=500,
        width=1000,
    )
    mem_spk_fig.update_xaxes(range=timestep_axis_range, dtick=timesteps // time_seconds)

    # Time (Seconds) vs Spike Counts Figure
    chunk_size = timesteps // time_seconds
    spk_rec = spk_rec.reshape(time_seconds, chunk_size, num_neurons).sum(axis=1)

    spk_counts_fig = go.Figure()
    for neuron_idx in range(num_neurons):
        spk_counts_fig.add_trace(
            go.Bar(
                x=[i + 0.5 for i in range(0, time_seconds)],  # Shift each bar by 0.5
                y=spk_rec[:, neuron_idx],
                name=f"Neuron {neuron_idx}",
                text=spk_rec[:, neuron_idx],  # Display counts
                textposition="inside",  # Place labels inside the bar
                marker=dict(color=colors[neuron_idx]),  # Set consistent color
            )
        )

    spk_counts_fig.update_layout(
        title="Spike Counts per Neuron Over Time",
        xaxis=dict(showgrid=True),
        xaxis_title="Time (Seconds)",
        yaxis_title="Spike Count",
        legend=dict(orientation="h", x=0, y=-0.1),
        barmode="stack",
        height=400,
        width=1000,
    )
    spk_counts_fig.update_xaxes(range=time_axis_range, dtick=1)

    # Display figures
    mem_spk_fig.show(config=config)
    spk_counts_fig.show(config=config)


def plot_predictions_and_labels(spk_rec, true_labels, time_axis_range=[0, 60], time_seconds=60):
    """
    Plots predictions vs ground truth and highlights incorrect classifications.

    Parameters:
        spk_rec (numpy.ndarray): Spike counts per neuron over reduced timesteps, shape (T, N).
        true_labels (numpy.ndarray): True labels for each timestep, shape (T,).
    """
    timesteps, num_neurons = spk_rec.shape
    chunk_size = timesteps // time_seconds
    preds = spk_rec.reshape(time_seconds, chunk_size, num_neurons).sum(axis=1)

    # Calculate predictions: neuron index with the maximum count in each timestep
    predictions = np.argmax(preds, axis=1)

    # Identify incorrect classifications
    incorrect_mask = predictions != true_labels

    # Plot predictions and ground truth
    timesteps = np.arange(preds.shape[0]) + 0.5  # Shift x-axis by 0.5

    fig = go.Figure()

    # Add true labels
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=true_labels,
            mode="lines+markers",
            name="Ground Truth",
            marker=dict(size=8, color="blue", symbol="square-open"),
            line=dict(color="blue"),
        )
    )

    # Add predictions
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=predictions,
            mode="lines+markers",
            name="Predictions",
            marker=dict(size=8, color="green", symbol="circle-open"),
            line=dict(color="green"),
        )
    )

    # Highlight incorrect predictions
    fig.add_trace(
        go.Scatter(
            x=timesteps[incorrect_mask],
            y=predictions[incorrect_mask],
            mode="markers",
            name="Incorrect Predictions",
            marker=dict(size=10, color="red", symbol="x"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Predictions vs Ground Truth",
        xaxis_title="Time (Seconds)",
        yaxis_title="Class",
        legend=dict(orientation="h", x=0, y=-0.8),
        height=250,
        width=1000,
    )
    fig.update_xaxes(range=time_axis_range, dtick=1)
    fig.update_yaxes(dtick=1)

    # Show plot
    fig.show(config=config)


def plot_correctness_matrix(y_locals, y_preds):

    # Set datatypes to int
    y_locals = y_locals.astype(int)
    y_preds = y_preds.astype(int)

    # Create a correctness mask: 1 if correct, 0 otherwise
    correct_mask = (y_locals == y_preds).astype(int)

    # Create a combined label for ground truth and predictions
    labels = np.array(
        [[f"{pred}/{gt}" for gt, pred in zip(row_gt, row_pred)] for row_gt, row_pred in zip(y_locals, y_preds)]
    )

    # Create a heatmap for correctness with labels
    fig = go.Figure(
        data=go.Heatmap(
            z=correct_mask,
            colorscale=[[0, "red"], [1, "green"]],  # Red for incorrect, green for correct
            showlegend=False,
            showscale=False,
            text=labels,
            texttemplate="%{text}",  # Format to show the text
            hoverinfo="y+text",  # Show both the correctness and the labels on hover
        )
    )

    fig.update_layout(
        title="Correctness of Predictions/Ground Truth",
        xaxis_title="Time (s)",
        yaxis_title="Samples",
        xaxis=dict(tickmode="linear"),
        yaxis=dict(tickmode="linear"),
        height=1200,
        width=1200,
    )

    fig.show(config=config)


def plot_confusion_matrix(y_locals, y_preds, class_labels):
    """
    Plots a confusion matrix heatmap.

    Parameters:
        y_locals (numpy.ndarray): Ground truth labels.
        y_preds (numpy.ndarray): Predicted labels.
        class_labels (dict): A dictionary where keys are class indices and values are their meanings as strings.
    """
    # Flatten arrays to compute confusion matrix
    y_locals_flat = y_locals.flatten()
    y_preds_flat = y_preds.flatten()

    # Compute confusion matrix
    labels = np.unique(np.concatenate([y_locals_flat, y_preds_flat]))
    conf_matrix = confusion_matrix(y_locals_flat, y_preds_flat, labels=labels)

    # Replace labels with their meanings
    labels_with_meanings = [class_labels.get(label, f"Class {label}") for label in labels]

    # Create a heatmap for the confusion matrix
    fig = go.Figure(
        data=go.Heatmap(
            z=conf_matrix,
            x=labels_with_meanings,
            y=labels_with_meanings,
            showscale=False,
            texttemplate="%{z:d}",
            textfont={"color": "white"},  # Make text white for better visibility
            hovertemplate="Ground Truth: %{x}<br>Prediction: %{y}<br>Count: %{z}<extra></extra>",
            zhoverformat=".0f"
        )
    )

    fig.update_layout(
        title={
            'text': "Confusion Matrix",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Predictions",
        yaxis_title="Ground Truth",
        xaxis=dict(
            tickmode="array", 
            tickvals=labels_with_meanings,
            tickangle=45  
        ),
        yaxis=dict(
            tickmode="array", 
            tickvals=labels_with_meanings
        ),
        height=800,
        width=800,
    )
    fig.show(config=config)

def plot_confusion_colored_matrix(y_true, y_pred, class_labels, colors):
    """
    Plots a confusion matrix heatmap with a custom color matrix.

    Parameters:
        y_true (numpy.ndarray): Ground truth labels.
        y_pred (numpy.ndarray): Predicted labels.
        class_labels (dict): A dictionary mapping class indices to their names.
        colors (list of list): A color matrix corresponding to the confusion matrix cells.
    """
    # Flatten arrays to compute confusion matrix
    y_locals_flat = y_true.flatten()
    y_preds_flat = y_pred.flatten()

    # Compute confusion matrix
    labels = np.unique(np.concatenate([y_locals_flat, y_preds_flat]))
    conf_matrix = confusion_matrix(y_locals_flat, y_preds_flat, labels=labels)

    # Ensure colors matrix matches the shape of the confusion matrix
    if len(colors) != conf_matrix.shape[0] or len(colors[0]) != conf_matrix.shape[1]:
        raise ValueError("The provided color matrix does not match the shape of the confusion matrix.")

    # Replace labels with their meanings
    labels_with_meanings = [class_labels.get(label, f"Class {label}") for label in labels]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw each cell with the appropriate color
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            rect = plt.Rectangle((j, i), 1, 1, facecolor=colors[i][j], edgecolor='black')
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, str(conf_matrix[i, j]), 
                    ha="center", va="center", fontsize=14, 
                    color="white" if colors[i][j] != "white" else "black")

    # Set axis labels and ticks
    ax.set_xticks(np.arange(len(labels_with_meanings)) + 0.5)
    ax.set_yticks(np.arange(len(labels_with_meanings)) + 0.5)
    ax.set_xticklabels(labels_with_meanings, rotation=45, ha="right")
    ax.set_yticklabels(labels_with_meanings)
    ax.set_xlim(0, conf_matrix.shape[1])
    ax.set_ylim(0, conf_matrix.shape[0])

    # Adjusting tick positions
    ax.set_xticks(np.arange(len(labels_with_meanings)), minor=True)
    ax.set_yticks(np.arange(len(labels_with_meanings)), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)

    # Labels
    ax.set_xlabel("Predictions")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")

    plt.show()