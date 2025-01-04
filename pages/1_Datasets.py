import streamlit as st
from utils.SpikingDataset import SpikingDataset
from utils.helpers import get_datasets_dirs
from utils.streamlit_helpers import display_video

datasets_dirs = get_datasets_dirs()

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Datasets")

# Select a dataset from the list
selected_dataset_name = st.selectbox("Select an Event Dataset:", list(datasets_dirs.keys()))

if selected_dataset_name:
    dataset = SpikingDataset(root_dir=datasets_dirs[selected_dataset_name])
    st.write(f"Number of videos: {len(dataset)}")

    # Input for searching/filtering by folder name
    search_query = st.text_input("Search")

    filtered_indices = []
    for index in range(len(dataset)):
        if search_query.lower() in dataset.get_folder_name(index).lower():
            filtered_indices.append(index)

    if filtered_indices:
        st.write(f"Number of videos after filtering: {len(filtered_indices)}")

        with st.form("checkbox_form", clear_on_submit=False):
            # Create columns for displaying videos in a grid
            cols = st.columns(3)
            for i, index in enumerate(filtered_indices):
                with cols[i % 3]:
                    display_video(dataset.get_video_path(index))
                    st.caption(dataset.get_folder_name(index))
                    is_fall = True if dataset.get_label(index) == 1 else False
                    # st.checkbox("Fall Label (Tick if Fall)", key=f"dataset-checkbox-{index}", value=is_fall)

            submitted = st.form_submit_button("Save Labels", type="primary")
            if submitted:
                for i, index in enumerate(filtered_indices):
                    correct_label = 1 if st.session_state[f"dataset-checkbox-{index}"] else 0
                    dataset.edit_label(index, correct_label)

                dataset.save_labels()
                st.write("Labels saved successfully.")
    else:
        st.write("No videos found for the given folder name.")
