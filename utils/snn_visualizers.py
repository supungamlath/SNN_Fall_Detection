import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp


def visualize_snn_output(mem_rec, spk_rec, timestep_range=None, time_seconds=60):
    """
    Visualizes the membrane potentials and spikes over time.

    Parameters:
        mem_rec (numpy.ndarray): Membrane potentials of shape (T, N).
        spk_rec (numpy.ndarray): Spikes of shape (T, N).
    """
    # Ensure the inputs are of the expected shape
    assert mem_rec.shape == spk_rec.shape, "mem_rec and spk_rec must have the same shape"

    timesteps, num_neurons = mem_rec.shape

    if not timestep_range:
        timestep_range = [0, timesteps]

    time = list(range(timesteps))  # Define the shared time axis

    # Membrane Potentials Figure
    mem_fig = go.Figure()
    for neuron_idx in range(num_neurons):
        mem_fig.add_trace(go.Scatter(x=time, y=mem_rec[:, neuron_idx], mode="lines", name=f"Neuron {neuron_idx}"))

    mem_fig.update_layout(
        title="Membrane Potentials Over Timesteps",
        xaxis_title="Time (Timesteps)",
        yaxis_title="Membrane Potential",
        legend=dict(orientation="h", x=0, y=-0.1),
        height=400,
        width=1000,
    )
    mem_fig.update_xaxes(range=timestep_range, dtick=timesteps // time_seconds)

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
        legend=dict(orientation="h", x=0, y=-0.4),
        height=250,
        width=1000,
    )
    spk_fig.update_xaxes(range=timestep_range, dtick=timesteps // time_seconds)

    spk_fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(0, num_neurons)),
        ticktext=[f"Neuron {i}" for i in range(num_neurons)],
    )

    # Reduced Timesteps vs Spike Counts Figure
    chunk_size = timesteps // time_seconds
    spk_rec = spk_rec.reshape(time_seconds, chunk_size, num_neurons).sum(axis=1)

    bar_fig = go.Figure()
    for neuron_idx in range(num_neurons):
        bar_fig.add_trace(
            go.Bar(
                x=list(range(0, time_seconds)),
                y=spk_rec[:, neuron_idx],
                name=f"Neuron {neuron_idx}",
            )
        )

    bar_fig.update_layout(
        title="Spike Counts per Neuron Over Time",
        xaxis_title="Time (Seconds)",
        yaxis_title="Spike Count",
        legend=dict(orientation="h", x=0, y=-0.1),
        barmode="stack",
        height=400,
        width=1000,
    )
    bar_fig.update_xaxes(range=[0, time_seconds], dtick=1)

    # Display figures
    mem_fig.show()
    spk_fig.show()
    bar_fig.show()


def plot_predictions_and_labels(spk_rec, true_labels, time_seconds=60):
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
    timesteps = np.arange(preds.shape[0])

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
        xaxis_title="Reduced Timesteps",
        yaxis_title="Class",
        legend=dict(orientation="h", x=0, y=-0.8),
        height=250,
        width=1000,
    )
    fig.update_xaxes(range=[0, time_seconds], dtick=1)

    # Show plot
    fig.show()


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
        subplot_titles=[
            f"Timestamp: {i * group_size} - {(i+1) * group_size}<br> Label: {int(labels[i])}" for i in range(timesteps)
        ],
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
                colorbar=dict(len=1.0, title="Intensity") if i == 0 else None,
                zmin=zmin,
                zmax=zmax,
            ),
            row=1,
            col=i + 1,
        )

    # Update layout
    fig.update_layout(
        title="Visualizing Vector Over Timesteps",
        height=300,
        width=timesteps * 250,
    )
    fig.update_annotations(font_size=14)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()


def visualize_input_grid(data, time_seconds=60, columns=10):
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

    fig.show()
