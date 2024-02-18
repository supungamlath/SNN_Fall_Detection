import streamlit as st
import torch
from utils.SpikingDataLoader import SpikingDataLoader
from utils.SpikingDataset import SpikingDataset
from utils.Trainer import Trainer
from utils.visualization import live_plot
from pages.helpers import datasets_dirs

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Training")


model_name = st.session_state.get("model_name", None)
st.write(f"### Selected Model: {model_name}")

if model_name:

    st.write("New Training Run")

    selected_dataset = st.selectbox("Datasets", list(datasets_dirs.keys()))
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
            dataset = SpikingDataset(
                root_dir=[datasets_dirs[selected_dataset]],
                max_time=model_params["max_time"],
                nb_steps=model_params["nb_steps"],
            )
            train_dataset, test_dataset = dataset.random_split(
                test_size=train_test_ratio, shuffle=True
            )
            train_loader = SpikingDataLoader(
                train_dataset, batch_size=model_params["batch_size"], shuffle=True
            )
            test_loader = SpikingDataLoader(
                test_dataset, batch_size=model_params["batch_size"], shuffle=False
            )

        with st.spinner("Training the model..."):
            loss_hist, train_accuracy_hist = trainer.train(
                train_loader, nb_epochs=nb_epochs, lr=learning_rate
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
                    f"### Training accuracy: {trainer.compute_accuracy(test_loader)}"
                )
                st.write(test_acc)

else:
    st.write("Please select a model first.")
