import os
import shlex
import subprocess
import streamlit as st
import torch

from utils.data_loader import EventsDataset, sparse_data_generator_from_hdf5_spikes
from utils.visualization import plot_voltage_traces

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Labelling")

model_name = st.session_state.get("model_name", None)
st.write(f"### Selected Model: {model_name}")

event_datasets = {
    "UR Fall Dataset": "URFD",
    "HAR UP Fall Dataset": "HAR-UP",
}


# Function to display a single video
def display_video(video_path):
    path_parts = video_path.split("/")
    file_name = path_parts[-1].replace(".avi", ".mp4")
    output_path = f"./temp/{path_parts[-2]}/{file_name}"
    if not os.path.exists(output_path):
        os.makedirs(f"./temp/{path_parts[-2]}", exist_ok=True)
        ffmpeg_command = (
            f"ffmpeg -y -i {video_path} -vcodec libx264 {output_path} -loglevel quiet"
        )
        subprocess.run(shlex.split(ffmpeg_command))

    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)


def draw_row_traces(x_local, model, model_params):
    output, _ = model.forward(x_local.to_dense())
    two_maxims, _ = torch.max(output, 1)  # max over time
    _, model_preds = torch.max(two_maxims, 1)  # argmax over output units
    diff = torch.abs(two_maxims[:, 0] - two_maxims[:, 1])
    plot_voltage_traces(
        mem=output.detach().cpu().numpy(),
        diff=diff.detach().cpu().numpy(),
        labels=model_preds.detach().cpu().tolist(),
        dim=(1, model_params["batch_size"]),
        renderer=st.pyplot,
    )


if model_name is not None:
    # Select a dataset from the list
    selected_dataset = st.selectbox(
        "Select an Event Dataset:", list(event_datasets.keys())
    )
    model = st.session_state["model"]
    model_params = st.session_state["model_params"]

    with st.spinner("Loading dataset..."):
        dataset = EventsDataset(
            datasets=[event_datasets[selected_dataset]],
            max_time=model_params["max_time"],
            nb_steps=model_params["nb_steps"],
            batch_size=model_params["batch_size"],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    video_files = dataset.get_video_files()

    with st.form("checkbox_form", clear_on_submit=False):
        # Create columns for displaying videos in a grid
        row_counter = 0
        x_all, y_all = dataset.get_samples()
        for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
            x_all,
            y_all,
            batch_size=model_params["batch_size"],
            nb_steps=model_params["nb_steps"],
            nb_units=model_params["nb_inputs"],
            max_time=model_params["max_time"],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            shuffle=False,
        ):
            cols = st.columns(7)
            for i in range(7):
                with cols[i]:
                    index = row_counter * 7 + i
                    display_video(video_files[index])
                    caption = f"Label: {y_all[index]}"
                    st.caption(caption)

                    st.checkbox("Label Fall?", key=f"checkbox-{index}")

            draw_row_traces(x_local, model, model_params)
            st.divider()
            row_counter += 1

        submitted = st.form_submit_button("Save Labels")
        if submitted:
            for i in range(len(video_files)):
                correct_label = st.session_state[f"checkbox-{i}"]
                dataset.set_correct_label(i, correct_label)

            st.write("Saved labels successfully.")

else:
    st.write("Please select a model first.")
