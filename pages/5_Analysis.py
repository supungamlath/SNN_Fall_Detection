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

# Plot selection options
plot_loss_history = st.checkbox("Plot Train Loss History")
plot_train_accuracy_history = st.checkbox("Plot Train Accuracy History")
plot_test_accuracy_history = st.checkbox("Plot Test Accuracy History")

# Plot Train Loss History
if plot_loss_history:
    st.header("Train Loss History")
    fig = px.line()
    for model_name in df["model_name"].unique():
        model_data = df[df["model_name"] == model_name]
        for idx, row in model_data.iterrows():
            if row["train_loss_hist"]:
                fig.add_scatter(
                    x=list(range(1, len(row["train_loss_hist"]) + 1)),
                    y=row["train_loss_hist"],
                    mode="lines",
                    name=f"{model_name} (Run at {row['datetime']})",
                )
    st.plotly_chart(fig)

# Plot Train Accuracy History
if plot_train_accuracy_history:
    st.header("Train Accuracy History")
    fig = px.line()
    for model_name in df["model_name"].unique():
        model_data = df[df["model_name"] == model_name]
        for idx, row in model_data.iterrows():
            if row["train_accuracy_hist"]:
                fig.add_scatter(
                    x=list(range(1, len(row["train_accuracy_hist"]) + 1)),
                    y=row["train_accuracy_hist"],
                    mode="lines",
                    name=f"{model_name} (Run at {row['datetime']})",
                )
    st.plotly_chart(fig)

# Plot Test Accuracy History
if plot_test_accuracy_history:
    st.header("Test Accuracy History")
    fig = px.line()
    for model_name in df["model_name"].unique():
        model_data = df[df["model_name"] == model_name]
        for idx, row in model_data.iterrows():
            if row["test_accuracy_hist"]:
                fig.add_scatter(
                    x=list(range(1, len(row["test_accuracy_hist"]) + 1)),
                    y=row["test_accuracy_hist"],
                    mode="lines",
                    name=f"{model_name} (Run at {row['datetime']})",
                )
    st.plotly_chart(fig)

st.write("Select rows and metrics to compare across different models.")
