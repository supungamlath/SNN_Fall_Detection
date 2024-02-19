import os
import streamlit as st
import torch

from models.model import SNN
from utils.helpers import load_params, save_params

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Models")

# Define the directory where the models and their parameters are stored
models_dir = f"{os.environ['root_folder']}/models/saved"
models_parameters_file = os.path.join(models_dir, "models.json")
training_runs_file = os.path.join(models_dir, "training_runs.json")

# Load existing model parameters
model_params = load_params(models_parameters_file)
model_files = [f[:-4] for f in os.listdir(models_dir) if f.endswith(".pth")]

with st.expander("Create New Model"):
    video_presets = {
        "240x180": 240 * 180,
        "320x240": 320 * 240,
        "640x480": 640 * 480,
    }
    model_name = st.text_input("Model Name")
    video_dims = st.selectbox("Video Dimensions", video_presets.keys(), index=0)
    nb_inputs = video_presets[video_dims]
    nb_hidden = st.number_input(
        "Number of Hidden Units", min_value=1000, max_value=10000, value=2000
    )
    nb_outputs = st.number_input("Number of Outputs", min_value=1, value=2)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=7)
    max_time = st.number_input("Max Time", min_value=1, max_value=60, value=15)
    nb_steps = st.number_input(
        "Number of Steps", min_value=500, max_value=5000, value=1000
    )

    # Button to create a new model
    if st.button("Save Model"):
        model_params[model_name] = {
            "nb_inputs": nb_inputs,
            "nb_hidden": nb_hidden,
            "nb_outputs": nb_outputs,
            "batch_size": batch_size,
            "max_time": max_time,
            "nb_steps": nb_steps,
        }
        model = SNN(
            nb_inputs=nb_inputs,
            nb_hidden=nb_hidden,
            nb_outputs=nb_outputs,
            batch_size=batch_size,
            max_time=max_time,
            nb_steps=nb_steps,
        )
        model.save(os.path.join(models_dir, f"{model_name}.pth"))
        save_params(models_parameters_file, model_params)
        model_files.append(model_name)
        st.success(f"Model {model_name} saved successfully.")

# Dropdown to select saved models
selected_model = st.selectbox(
    "Select a saved model:", model_files, index=len(model_files) - 1
)

# Display model parameters before loading
if selected_model:
    if selected_model in model_params:
        st.write(f"Model Parameters for {selected_model}")
        params_string = "\n"
        for k, v in model_params[selected_model].items():
            params_string += f"{k}: {v} \n"
        params_string = f"```{params_string}```"
        st.markdown(params_string)
    else:
        st.write(f"No parameters found for {selected_model}")

# cols = st.columns(6)
# with cols[2]:
# Load button to load the model to Streamlit's session state
if st.button("Load Model"):
    if selected_model:
        model_path = os.path.join(models_dir, selected_model + ".pth")

        st.session_state["model_name"] = selected_model
        with st.spinner("Loading model..."):
            st.session_state["model"] = SNN.load(model_path)
            st.session_state["model_params"] = model_params[selected_model]
        st.success(f"Model {selected_model} loaded successfully.")
    else:
        st.error("Please select a model to load.")
# with cols[3]:
# Delete button to delete the selected model
if st.button("Delete Model", type="primary"):
    if selected_model:
        os.remove(os.path.join(models_dir, selected_model + ".pth"))
        del model_params[selected_model]
        save_params(models_parameters_file, model_params)
        model_files.remove(selected_model)
        training_runs = load_params(training_runs_file)
        if selected_model in training_runs:
            del training_runs[selected_model]
            save_params(training_runs_file, training_runs)
        st.success(f"Model {selected_model} deleted successfully.")
    else:
        st.error("Please select a model to delete.")
