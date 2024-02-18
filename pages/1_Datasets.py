import streamlit as st
import os
import subprocess
import shlex

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Datasets")

# Define the path to the video datasets
video_datasets_path = {
    "UR Fall Dataset": "./data/urfd-spiking-dataset-240",
    "HAR UP Fall Dataset": "./data/har-up-spiking-dataset-240",
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


# Select a dataset from the list
selected_dataset_name = st.selectbox(
    "Select an Event Dataset:", list(video_datasets_path.keys())
)

# Get the path of the selected dataset
selected_dataset_path = video_datasets_path[selected_dataset_name]

# Display videos as tiles with file names beneath them
if selected_dataset_path:
    # List all .avi files in the selected directory
    video_folders = [
        f
        for f in os.listdir(selected_dataset_path)
        if os.path.isdir(os.path.join(selected_dataset_path, f))
    ]

    if video_folders:
        # Create columns for displaying videos in a grid
        cols = st.columns(3)  # Adjust the number of columns as needed
        for index, video_folder in enumerate(video_folders):
            with cols[index % 3]:
                display_video(f"{selected_dataset_path}/{video_folder}/dvs-video.avi")
                st.caption(video_folder)
    else:
        st.write("No video files found in the selected dataset.")
