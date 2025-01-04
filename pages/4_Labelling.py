import streamlit as st
from utils.SpikingDataLoader import SpikingDataLoader
from utils.SpikingDataset import SpikingDataset
from utils.helpers import get_datasets_dirs, display_video, draw_row_traces

datasets_dirs = get_datasets_dirs()

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Labelling")

model_name = st.session_state.get("model_name", None)
st.write(f"### Selected Model: {model_name}")


if model_name is not None:
    # Select a dataset from the list
    selected_dataset = st.selectbox("Select an Event Dataset:", list(datasets_dirs.keys()))
    model = st.session_state["model"]
    model_params = st.session_state["model_params"]

    with st.spinner("Loading dataset..."):
        dataset = SpikingDataset(
            root_dir=datasets_dirs[selected_dataset],
            max_time=model_params["max_time"],
            nb_steps=model_params["nb_steps"],
        )
        dataloader = SpikingDataLoader(dataset, batch_size=model_params["batch_size"], shuffle=False)

    with st.form("checkbox_form", clear_on_submit=False):
        # Create columns for displaying videos in a grid
        row_counter = 0
        for x_local, y_local in dataloader:
            cols = st.columns(dataloader.batch_size)
            for i in range(dataloader.batch_size):
                with cols[i]:
                    index = row_counter * dataloader.batch_size + i
                    display_video(dataset.get_video_path(index))
                    is_fall = True if dataset.get_label(index) == 1 else False
                    st.checkbox("Is Fall?", key=f"labelling-checkbox-{index}", value=is_fall)

            draw_row_traces(x_local, model, model_params)
            st.divider()
            row_counter += 1

        submitted = st.form_submit_button("Save Labels", type="primary")
        if submitted:
            for i in range(len(dataset)):
                correct_label = 1 if st.session_state[f"labelling-checkbox-{i}"] else 0
                dataset.edit_label(i, correct_label)

            dataset.save_labels()
            st.write("Labels saved successfully.")

else:
    st.write("Please select a model first.")
