# Define the path to the video datasets
import json
import os
import shlex
import subprocess
import streamlit as st
import torch

from utils.visualization import plot_voltage_traces


datasets_dirs = {
    "UR Fall Dataset": f"{os.environ['root_folder']}/data/urfd-spiking-dataset-240",
    "HAR UP Fall Dataset": f"{os.environ['root_folder']}/data/har-up-spiking-dataset-240",
}


# Function to display a single video
def display_video(video_path, trim_time=15):
    # Split the path into directory and file name
    head, file_name = os.path.split(video_path)
    # Split off the extension and replace it
    base_name, _ = os.path.splitext(file_name)
    new_file_name = base_name + ".mp4"
    # Construct the new output path
    output_path = os.path.join(".", "temp", os.path.basename(head), new_file_name)

    # Check if the output path exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ffmpeg_command = f'ffmpeg -y -t {trim_time} -i "{video_path}" -vcodec libx264 "{output_path}" -loglevel quiet'
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


def print_loss_accuracy(loss_hist, train_accuracy_hist, test_accuracy_hist=None):
    st.write(f"Epoch: {len(train_accuracy_hist)}")
    markdown_str = (
        f"\nLoss: {loss_hist[-1]} \nTrain Set Accuracy: {train_accuracy_hist[-1]}\n"
    )
    if test_accuracy_hist:
        markdown_str += f"Test Set Accuracy: {test_accuracy_hist[-1]}\n"
    markdown_str = f"```{markdown_str}```"
    st.markdown(markdown_str)


# Function to save parameters to file
def save_params(file_dir, params):
    with open(file_dir, "w") as f:
        json.dump(params, f, indent=4)


# Function to load parameters from file
def load_params(file_dir):
    if os.path.exists(file_dir):
        with open(file_dir, "r") as f:
            params = json.load(f)
        return params
    else:
        return {}
