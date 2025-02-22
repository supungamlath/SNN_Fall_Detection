import json
import plotly.graph_objects as go
import random


def create_plot(data, title="Performance Over Time", x_label="Step", y_label="Value", legend_title="", max_epochs=None):
    """
    Create a Plotly line plot from JSON data.

    Args:
        data (list): List of dictionaries containing plot data
        title (str): Title for the plot
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        legend_title (str): Title for the legend
        max_epochs (int, optional): Maximum number of epochs to plot

    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    # Create the figure
    fig = go.Figure()

    # Define available marker symbols
    marker_symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "pentagon",
        "hexagon",
        "star",
        "hexagram",
        "star-triangle-up",
        "star-triangle-down",
        "star-square",
        "star-diamond",
        "diamond-tall",
        "diamond-wide",
    ]

    # Sort the data by name
    sorted_data = sorted(data, key=lambda x: int(x["name"]) if x["name"].isdigit() else x["name"])

    # Shuffle marker symbols to ensure random but non-repeating selection
    random.shuffle(marker_symbols)

    # Add traces for each dataset
    for i, trace in enumerate(sorted_data):
        # If max_epochs is set, limit the data points
        if max_epochs is not None:
            x_data = trace["x"][:max_epochs]
            y_data = trace["y"][:max_epochs]
        else:
            x_data = trace["x"]
            y_data = trace["y"]

        # Select marker symbol (cycling through if more traces than symbols)
        marker_symbol = marker_symbols[i % len(marker_symbols)]

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                name=trace["name"],
                mode="lines+markers",
                line=dict(width=trace.get("line", {}).get("width", 2)),
                marker=dict(symbol=marker_symbol, size=8),
                hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.4f}}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            title=dict(text=legend_title, side="top"),
        ),
        template="plotly_white",
        margin=dict(t=100, l=80, r=80, b=80),
    )

    # Update axes
    # Update axes with enhanced styling
    fig.update_xaxes(showgrid=True, gridwidth=1, mirror=True, ticks="outside", showline=True)
    fig.update_yaxes(showgrid=True, gridwidth=1, mirror=True, ticks="outside", showline=True)

    return fig


def load_and_plot(
    file_path, title="Performance Over Time", x_label="Step", y_label="Value", legend_title="", max_epochs=None
):
    """
    Load JSON data from file and create plot.

    Args:
        file_path (str): Path to JSON file
        title (str): Title for the plot
        x_label (str): Label for x-axis
        y_label (str): Label for y-axis
        legend_title (str): Title for the legend
        max_epochs (int, optional): Maximum number of epochs to plot

    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    # Load data from JSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Create and return the plot
    return create_plot(data, title, x_label, y_label, legend_title, max_epochs)


if __name__ == "__main__":
    # You can use this with either a JSON string or file

    fig = load_and_plot(
        "data/plots/dev _ f1_score.json",
        title="",
        x_label="Epoch",
        y_label="Validation F1 Score",
        legend_title="Number of Timesteps",
        max_epochs=10,
    )

    # Show the plot
    fig.show()
