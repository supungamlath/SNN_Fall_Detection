import streamlit as st
import torch
from utils.data_loader import EventsDataset

from utils.trainer import Trainer
from utils.visualization import live_plot

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Training")

# Define the path to the video datasets
event_datasets = {
    "UR Fall Dataset": "URFD",
    "HAR UP Fall Dataset": "HAR-UP",
}

model_name = st.session_state.get("model_name", None)
st.write(f"### Selected Model: {model_name}")

if model_name:

    st.write("New Training Run")

    selected_datasets = st.multiselect("Datasets", list(event_datasets.keys()))
    train_test_ratio = st.slider(
        "Train/Test Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.05
    )
    nb_epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=5)
    learning_rate = st.number_input(
        "Learning Rate", min_value=1e-5, max_value=1e-1, value=2e-4, format="%.5f"
    )

    if st.button("Start Training Run"):
        model = st.session_state["model"]
        model_params = st.session_state["model_params"]
        trainer = Trainer(model=model, graph_renderer=st.pyplot)
        with st.spinner("Loading datasets..."):
            dataset = EventsDataset(
                datasets=[event_datasets[dataset] for dataset in selected_datasets],
                max_time=model_params["max_time"],
                nb_steps=model_params["nb_steps"],
                batch_size=model_params["batch_size"],
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=0.25)
        with st.spinner("Training the model..."):
            loss_hist, train_accuracy_hist = trainer.train(
                x_train, y_train, nb_epochs=nb_epochs, lr=learning_rate
            )
        with st.spinner("Saving the model..."):
            torch.save(model, f"./models/saved/{model_name}.pth")

        col1, col2 = st.columns(2)
        with col1:
            live_plot(
                loss_hist, title="Loss History", y_label="Loss", renderer=st.pyplot
            )
        with col2:
            live_plot(
                train_accuracy_hist,
                title="Training Accuracy",
                y_label="Accuracy",
                renderer=st.pyplot,
            )
        st.success("Training run completed successfully.")

        st.write("### Evaluation")

        if st.button("Start Evaluation"):
            with st.spinner("Evaluating test set accuracy..."):
                test_acc = (
                    f"### Training accuracy: {trainer.compute_accuracy(x_test, y_test)}"
                )
                st.write(test_acc)

else:
    st.write("Please select a model first.")
