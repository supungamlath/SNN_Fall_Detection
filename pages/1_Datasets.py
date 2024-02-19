import streamlit as st
from utils.SpikingDataset import SpikingDataset
from utils.helpers import datasets_dirs, display_video

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Datasets")

# Select a dataset from the list
selected_dataset_name = st.selectbox(
    "Select an Event Dataset:", list(datasets_dirs.keys())
)

# Display videos as tiles with file names beneath them
if selected_dataset_name:
    dataset = SpikingDataset(root_dir=datasets_dirs[selected_dataset_name])
    st.write(f"Number of videos: {len(dataset)}")
    # Create columns for displaying videos in a grid
    cols = st.columns(3)
    for index in range(len(dataset)):
        with cols[index % 3]:
            display_video(dataset.get_video_path(index))
            st.caption(dataset.get_folder_name(index))
