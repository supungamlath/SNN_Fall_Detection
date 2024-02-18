import json
import os
import streamlit as st
import torch

from models.model import SNN

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Models")

# Define the directory where the models and their parameters are stored
models_dir = "./models/saved"
models_parameters_file = os.path.join(models_dir, "models.json")


# Function to save model parameters to JSON
def save_model_params(params):
    with open(models_parameters_file, "w") as f:
        json.dump(params, f, indent=4)


# Function to load model parameters from JSON
def load_model_params():
    if os.path.exists(models_parameters_file):
        with open(models_parameters_file, "r") as f:
            params = json.load(f)
        return params
    else:
        return {}


# Load existing model parameters
model_params = load_model_params()

# Dropdown to select saved models
model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
selected_model_file = st.selectbox(
    "Select a saved model:", model_files, index=len(model_files) - 1
)

# Display model parameters before loading
if selected_model_file:
    model_name = selected_model_file[:-4]  # Remove the .pth extension
    if model_name in model_params:
        st.write(f"Model Parameters for {model_name}")
        for k, v in model_params[model_name].items():
            st.write(f"`{k}: {v}`")
    else:
        st.write("No parameters found for the selected model.")

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
    if st.button("Create New Model"):
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
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        torch.save(model, os.path.join(models_dir, f"{model_name}.pth"))
        save_model_params(model_params)
        selected_model_file = f"{model_name}.pth"
        st.success(f"Model {model_name} created successfully.")


# Load button to load the model to Streamlit's session state
if st.button("Load Model"):
    if selected_model_file:
        model_path = os.path.join(models_dir, selected_model_file)
        model_name = selected_model_file[:-4]

        st.session_state["model_name"] = model_name
        with st.spinner("Loading model..."):
            st.session_state["model"] = torch.load(model_path)
            st.session_state["model_params"] = model_params[model_name]
        st.success(f"Model {model_name} loaded successfully.")
    else:
        st.error("Please select a model to load.")
