import os
import pandas as pd
import streamlit as st
import plotly.express as px
from utils.helpers import (
    load_params,
    training_json_to_dataframe,
    models_info_json_to_dataframe,
)


# Define the directory where the models and their parameters are stored
models_dir = f"{os.environ['root_folder']}/models/saved"
models_parameters_file = os.path.join(models_dir, "models.json")
training_runs_file = os.path.join(models_dir, "training_runs.json")


# Create DataFrames
training_data = load_params(training_runs_file)
models_info = load_params(models_parameters_file)
training_df = training_json_to_dataframe(training_data)
models_info_df = models_info_json_to_dataframe(models_info)

# Merge the DataFrames on model_name
df = pd.merge(models_info_df, training_df, on="model_name")

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Analysis")

# Display the dataframe
st.dataframe(df)

# Filter options
model_names = st.multiselect("Select Model Names", df["model_name"].unique())
datasets = st.multiselect("Select Datasets", df["dataset"].unique())
if model_names:
    df = df[df["model_name"].isin(model_names)]
if datasets:
    df = df[df["dataset"].isin(datasets)]

# Legend selection option
legends = {
    "Model Name": "model_name",
    "Dataset": "dataset",
    "Memory Time Constant": "tau_mem",
    "Synaptic Time Constant": "tau_syn",
    "Layer Sizes": "snn_layers",
}
legend_option = st.selectbox("Select Legend for Plots", legends.keys())

# Plot selection options
with st.expander("Select Plots to Display", expanded=True):
    cols = st.columns(2)
    with cols[0]:
        plot_train_accuracy_history = st.checkbox("Train Set Accuracy")
        plot_train_precision_history = st.checkbox("Train Set Precision")
        plot_train_recall_history = st.checkbox("Train Set Recall")
        plot_train_f1_score_history = st.checkbox("Train Set F1 Score")
        plot_train_loss_history = st.checkbox("Train Set Loss")
    with cols[1]:
        plot_test_accuracy_history = st.checkbox("Test Set Accuracy")
        plot_test_precision_history = st.checkbox("Test Set Precision")
        plot_test_recall_history = st.checkbox("Test Set Recall")
        plot_test_f1_score_history = st.checkbox("Test Set F1 Score")


# Common plot format for all plots
def plot_metrics_format(title, list_column_name):
    st.header(title)
    fig = px.line()
    sorted_df = df.sort_values(by=legends[legend_option])
    for model_name in sorted_df["model_name"].unique():
        model_data = sorted_df[sorted_df["model_name"] == model_name]
        for idx, row in model_data.iterrows():
            if row.get(list_column_name):
                fig.add_scatter(
                    x=list(range(1, len(row[list_column_name]) + 1)),
                    y=row[list_column_name],
                    mode="lines",
                    name=f"{legend_option} - {row[legends[legend_option]]}",
                )
    st.plotly_chart(fig)


if plot_train_accuracy_history:
    plot_metrics_format("Train Set Accuracy History", "train_accuracy_hist")
if plot_train_precision_history:
    plot_metrics_format("Train Set Precision History", "train_precision_hist")
if plot_train_recall_history:
    plot_metrics_format("Train Set Recall History", "train_recall_hist")
if plot_train_f1_score_history:
    plot_metrics_format("Train Set F1 Score History", "train_f1_score_hist")
if plot_train_loss_history:
    plot_metrics_format("Train Set Loss History", "train_loss_hist")

if plot_test_accuracy_history:
    plot_metrics_format("Test Set Accuracy History", "test_accuracy_hist")
if plot_test_precision_history:
    plot_metrics_format("Test Set Precision History", "test_precision_hist")
if plot_test_recall_history:
    plot_metrics_format("Test Set Recall History", "test_recall_hist")
if plot_test_f1_score_history:
    plot_metrics_format("Test Set F1 Score History", "test_f1_score_hist")

st.write("Select rows and metrics to compare across different models.")
