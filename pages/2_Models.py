import os
import streamlit as st

from models.SpikingNN import SpikingNN
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
    snn_layers = []
    video_dims = st.selectbox("Neurons in Input Layer (Video Dimensions)", video_presets.keys(), index=0)
    nb_inputs = video_presets[video_dims]
    snn_layers.append(nb_inputs)

    num_hidden_layers = st.number_input("Number of Hidden Layers", min_value=1, max_value=10, value=1)

    for i in range(num_hidden_layers):
        nb_hidden = st.number_input(
            f"Neurons in Hidden Layer {i + 1}",
            min_value=1000,
            max_value=10000,
            value=2000,
            key=f"hidden_layer_{i}",
        )
        snn_layers.append(nb_hidden)

    # Output layer
    snn_layers.append(2)

    tau_mem = st.number_input("Membrane Time Constant (miliseconds)", min_value=10, max_value=500, value=100)
    tau_syn = st.number_input("Synaptic Time Constant (miliseconds)", min_value=10, max_value=500, value=50)
    nb_steps = st.number_input("Number of Time steps", min_value=500, max_value=10000, value=3000)
    time_step = st.number_input("Time Step (miliseconds)", min_value=1, max_value=100, value=20)

    # Button to create a new model
    if st.button("Save Model"):
        model_params[model_name] = {
            "snn_layers": snn_layers,
            "nb_steps": nb_steps,
            "time_step": time_step * 1e-3,
            "tau_mem": tau_mem * 1e-3,
            "tau_syn": tau_syn * 1e-3,
            "max_time": 60,
            "batch_size": 7,
        }
        model = SpikingNN(
            layer_sizes=snn_layers,
            nb_steps=nb_steps,
            time_step=time_step * 1e-3,
            tau_mem=tau_mem * 1e-3,
            tau_syn=tau_syn * 1e-3,
        )
        model.save(os.path.join(models_dir, f"{model_name}.pth"))
        save_params(models_parameters_file, model_params)
        model_files.append(model_name)
        st.success(f"Model {model_name} saved successfully.")

# Dropdown to select saved models
selected_model = st.selectbox("Select a saved model:", model_files, index=len(model_files) - 1)

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

cols = st.columns(2)
with cols[0]:
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
with cols[1]:
    # Delete button to delete the selected model
    if st.button("Delete Model", type="primary"):
        if selected_model:
            model_files.remove(selected_model)
            os.remove(os.path.join(models_dir, selected_model + ".pth"))
            if selected_model in model_params:
                del model_params[selected_model]
                save_params(models_parameters_file, model_params)
            training_runs = load_params(training_runs_file)
            if selected_model in training_runs:
                del training_runs[selected_model]
                save_params(training_runs_file, training_runs)
            st.success(f"Model {selected_model} deleted successfully.")
        else:
            st.error("Please select a model to delete.")
