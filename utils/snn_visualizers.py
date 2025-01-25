import numpy as np
from snntorch import spikeplot as splt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp


# From Open Neuromorphic Seminar - Hands-On with snnTorch by Jason K. Eshraghian (www.ncg.ucsc.edu)
def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, num_steps, title):
    # Generate Plots
    fig, ax = plt.subplots(3, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.4]})

    # Plot input spikes
    splt.raster(spk_in[:, 0], ax[0], s=0.03, c="black")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_title(title)

    # Plot hidden layer spikes
    splt.raster(spk1_rec.reshape(num_steps, -1), ax[1], s=0.05, c="black")
    ax[1].set_ylabel("Hidden Layer")

    # Plot output spikes
    splt.raster(spk2_rec.reshape(num_steps, -1), ax[2], c="black", marker="|")
    ax[2].set_ylabel("Output Spikes")
    ax[2].set_ylim([0, 10])

    plt.show()


# From Open Neuromorphic Seminar - Hands-On with snnTorch by Jason K. Eshraghian (www.ncg.ucsc.edu)
def dvs_animator(spike_data):
    fig, ax = plt.subplots()
    anim = splt.animator((spike_data[:, 0] + spike_data[:, 1]), fig, ax)
    return anim


def visualize_snn_output(mem_rec, spk_rec, time_range=None, time_ticks=60):
    """
    Visualizes the membrane potentials and spikes over time.

    Parameters:
        mem_rec (numpy.ndarray): Membrane potentials of shape (T, N).
        spk_rec (numpy.ndarray): Spikes of shape (T, N).
    """
    # Ensure the inputs are of the expected shape
    assert mem_rec.shape == spk_rec.shape, "mem_rec and spk_rec must have the same shape"

    timesteps, num_neurons = mem_rec.shape

    if not time_range:
        time_range = [0, timesteps]

    time = list(range(timesteps))  # Define the shared time axis

    # Membrane Potentials Figure
    mem_fig = go.Figure()
    for neuron_idx in range(num_neurons):
        mem_fig.add_trace(go.Scatter(x=time, y=mem_rec[:, neuron_idx], mode="lines", name=f"Neuron {neuron_idx}"))

    mem_fig.update_layout(
        title="Membrane Potentials Over Time",
        xaxis_title="Time (Timesteps)",
        yaxis_title="Membrane Potential",
        legend=dict(orientation="h", x=0, y=-0.2),
        height=400,
        width=1000,
    )
    mem_fig.update_xaxes(range=time_range, dtick=timesteps // time_ticks)

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
        title="Spikes Over Time",
        xaxis_title="Time (Timesteps)",
        yaxis_title="Neuron Index",
        legend=dict(orientation="h", x=0, y=-0.5),
        height=250,
        width=1000,
    )
    spk_fig.update_xaxes(range=time_range, dtick=timesteps // time_ticks)

    spk_fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(0, num_neurons)),
        ticktext=[f"Neuron {i}" for i in range(num_neurons)],
    )

    # Display figures
    mem_fig.show()
    spk_fig.show()


def visualize_events(
    data: np.ndarray,
    labels: np.ndarray,
    plot_timesteps: int = None,
):
    """
    Visualizes a 3D vector (timesteps, height, width) using Plotly with an option
    to reduce the timesteps and display averaged frames. Ensures all plots share the same intensity scale.

    Parameters:
        data (np.ndarray): Input data of shape (timesteps, height, width).
        labels (np.ndarray): Labels for each timestep.
        plot_timesteps (int): Number of timesteps to reduce to. If None, all timesteps are used.
    """
    timesteps, width, height = data.shape

    # Reshape and sum events to reduce timesteps if specified
    if plot_timesteps is not None:
        assert timesteps % plot_timesteps == 0, "Timesteps must be divisible by plot_timesteps."
        group_size = timesteps // plot_timesteps
        data = data.reshape(plot_timesteps, group_size, width, height).sum(axis=1)

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
                colorscale="Viridis",
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

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()


def visualize_input_grid(data, plot_timesteps=60, columns=10):
    timesteps, height, width = data.shape
    # Reshape and average data to reduce timesteps
    if plot_timesteps is not None:
        assert timesteps % plot_timesteps == 0, "Timesteps must be divisible by plot_timesteps."
        group_size = timesteps // plot_timesteps
        data = data.reshape(plot_timesteps, group_size, height, width).mean(axis=1)

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
