import os
import shlex
import subprocess
import streamlit as st
import torch

from utils.visualization import plot_voltage_traces


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


def print_loss_accuracy(train_metrics_hist, dev_metrics_hist=None):
    st.write(f"Epoch: {len(train_metrics_hist)}")
    train_metrics = train_metrics_hist[-1]
    markdown_str = f"\nTrain Loss: {train_metrics['loss']} \nTrain Set Accuracy: {train_metrics['accuracy']}\n"
    if dev_metrics_hist:
        dev_metrics = dev_metrics_hist[-1]
        markdown_str += f"Dev Set Accuracy: {dev_metrics['accuracy']}\n"
    markdown_str = f"```{markdown_str}```"
    st.markdown(markdown_str)
