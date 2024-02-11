from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objs as go
from plotly.offline import iplot

label_names = {0: "Non-Fall", 1: "Fall"}


def plot_voltage_traces(mem, spk=None, labels=None, dim=(1, 7), spike_height=5):
    plt.figure(figsize=(14, 2))
    gs = GridSpec(*dim)

    if spk is not None:
        dat = 1.0 * mem
        dat[spk > 0.0] = spike_height
    else:
        dat = mem

    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)

        ax.plot(dat[i])
        if labels is not None:
            ax.text(
                0.5,
                -0.5,
                label_names[labels[i]],
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        # ax.axis("off")
    plt.tight_layout()
    plt.show()


def live_plot(loss):
    if len(loss) == 1:
        return
    ax = plt.figure(figsize=(3, 2), dpi=150).gca()
    ax.plot(range(1, len(loss) + 1), loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.get_major_locator().set_params(integer=True)
    sns.despine()
    plt.show()


def plot_neuron_activity(data, title, x_title, y_title):
    fig = go.Figure(
        data=go.Heatmap(z=data, colorscale="Greys", colorbar=dict(title="Spike Count"))
    )

    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=x_title),
        yaxis=dict(title=y_title, scaleanchor="x", scaleratio=1),
        template="plotly_white",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
    )

    iplot(fig)
