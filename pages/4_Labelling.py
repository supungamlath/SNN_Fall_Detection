import streamlit as st
from utils.SpikingDataLoader import SpikingDataLoader
from utils.SpikingDataset import SpikingDataset
from pages.helpers import datasets_dirs, display_video, draw_row_traces

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Labelling")

model_name = st.session_state.get("model_name", None)
st.write(f"### Selected Model: {model_name}")


if model_name is not None:
    # Select a dataset from the list
    selected_dataset = st.selectbox(
        "Select an Event Dataset:", list(datasets_dirs.keys())
    )
    model = st.session_state["model"]
    model_params = st.session_state["model_params"]

    with st.spinner("Loading dataset..."):
        dataset = SpikingDataset(
            root_dir=[datasets_dirs[selected_dataset]],
            max_time=model_params["max_time"],
            nb_steps=model_params["nb_steps"],
        )
        dataloader = SpikingDataLoader(
            dataset, batch_size=model_params["batch_size"], shuffle=False
        )

    with st.form("checkbox_form", clear_on_submit=False):
        # Create columns for displaying videos in a grid
        row_counter = 0
        for x_local, y_local in dataloader:
            cols = st.columns(dataloader.batch_size)
            for i in range(dataloader.batch_size):
                with cols[i]:
                    index = row_counter * dataloader.batch_size + i
                    display_video(dataset.get_video_path(index))
                    caption = f"Label: {dataset.get_label(index)}"
                    st.caption(caption)

                    st.checkbox("Label Fall?", key=f"checkbox-{index}")

            draw_row_traces(x_local, model, model_params)
            st.divider()
            row_counter += 1

        submitted = st.form_submit_button("Save Labels")
        if submitted:
            for i in range(len(dataset)):
                correct_label = st.session_state[f"checkbox-{i}"]
                dataset.edit_label(i, correct_label)

            dataset.save_labels()
            st.write("Saved labels successfully.")

else:
    st.write("Please select a model first.")
