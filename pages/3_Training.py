from datetime import datetime
import os
import streamlit as st
import torch
from utils.SpikingDataLoader import SpikingDataLoader
from utils.SpikingDataset import SpikingDataset
from utils.Trainer import Trainer
from utils.visualization import live_plot_plotly
from utils.helpers import datasets_dirs, load_params, print_loss_accuracy, save_params

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Training")

# Define the directory where the previous training runs are stored
training_runs_file = f"{os.environ['root_folder']}/models/saved/training_runs.json"

# Load previous training runs
training_params = load_params(training_runs_file)

model_name = st.session_state.get("model_name", None)
st.write(f"### Selected Model: {model_name}")

if model_name:
    with st.expander("Previous Training Runs"):
        if model_name in training_params:
            previous_training_runs = training_params[model_name]
            for run in previous_training_runs:
                st.write(f"Training Run - {run['datetime']}")
                params_string = (
                    f"\nDataset: {run['dataset']} \n"
                    f"Train/Test Ratio: {run['train_test_ratio']} \n"
                    f"Number of Epochs: {run['nb_epochs']} \n"
                    f"Learning Rate: {run['learning_rate']} \n"
                )
                params_string = f"```{params_string}```"
                st.markdown(params_string)
                if "train_accuracy_hist" in run:
                    live_plot_plotly(
                        run["train_accuracy_hist"],
                        run["test_accuracy_hist"],
                        title="Model Accuracy",
                        y_label="Accuracy",
                        renderer=st.plotly_chart,
                    )
                else:
                    st.write("`No training history available.`")

                if st.button("Delete Training Run", key=run["datetime"]):
                    previous_training_runs.remove(run)
                    save_params(training_runs_file, training_params)
                st.divider()
        else:
            st.write("No training runs yet.")

    st.write("New Training Run")
    start_training_disabled = False
    selected_dataset = st.selectbox("Dataset", list(datasets_dirs.keys()))
    model_params = st.session_state["model_params"]
    dataset = SpikingDataset(
        root_dir=datasets_dirs[selected_dataset],
        max_time=model_params["max_time"],
        nb_steps=model_params["nb_steps"],
    )

    train_test_ratio = st.slider(
        "Train/Test Ratio", min_value=0.0, max_value=1.0, value=0.20, step=0.01
    )
    nb_test_samples = int(len(dataset) * train_test_ratio)
    nb_train_samples = len(dataset) - nb_test_samples
    cols = st.columns(6)
    with cols[2]:
        st.write(f"Train Samples: {nb_train_samples}")
    with cols[3]:
        st.write(f"Test Samples: {nb_test_samples}")
    if (
        nb_test_samples % model_params["batch_size"] != 0
        or nb_train_samples % model_params["batch_size"] != 0
    ):
        st.error(
            f"Sample sizes should be divisible by batch size {model_params['batch_size']}"
        )
        start_training_disabled = True
    else:
        start_training_disabled = False

    nb_epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=5)
    learning_rate = st.number_input(
        "Learning Rate", min_value=1e-5, max_value=1e-1, value=2e-4, format="%.5f"
    )
    evaluate_on_epoch = st.checkbox("Evaluate test samples on each epoch", value=True)

    if st.button(
        "Start Training Run", type="primary", disabled=start_training_disabled
    ):

        if model_name not in training_params:
            training_params[model_name] = []
        training_params[model_name].append(
            {
                "datetime": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "dataset": selected_dataset,
                "train_test_ratio": train_test_ratio,
                "nb_epochs": nb_epochs,
                "learning_rate": learning_rate,
            }
        )
        save_params(training_runs_file, training_params)

        with st.spinner("Loading datasets..."):
            train_dataset, test_dataset = dataset.random_split(
                test_size=train_test_ratio, shuffle=True
            )
            train_loader = SpikingDataLoader(
                train_dataset, batch_size=model_params["batch_size"], shuffle=False
            )
            test_loader = SpikingDataLoader(
                test_dataset, batch_size=model_params["batch_size"], shuffle=False
            )

        with st.spinner("Training the model..."):
            model = st.session_state["model"]
            trainer = Trainer(model=model)

            histories = trainer.train(
                train_loader,
                nb_epochs=nb_epochs,
                lr=learning_rate,
                evaluate_loader=test_loader if evaluate_on_epoch else None,
                callback_fn=print_loss_accuracy,
            )
            training_params[model_name][-1]["loss_hist"] = histories[0]
            training_params[model_name][-1]["train_accuracy_hist"] = histories[1]
            training_params[model_name][-1]["test_accuracy_hist"] = histories[2]
            save_params(training_runs_file, training_params)

        with st.spinner("Saving the model..."):
            torch.save(
                model, f"{os.environ['root_folder']}/models/saved/{model_name}.pth"
            )

        col1, col2 = st.columns(2)
        with col1:
            live_plot_plotly(
                histories[0],
                title="Training Loss",
                y_label="Loss",
                renderer=st.plotly_chart,
            )
        with col2:
            live_plot_plotly(
                histories[1],
                histories[2],
                title="Model Accuracy",
                y_label="Accuracy",
                renderer=st.plotly_chart,
            )
        st.success("Training run completed successfully.")

        if not evaluate_on_epoch:
            with st.spinner("Evaluating test set accuracy..."):
                test_acc = f"```\nTest Set Accuracy: {trainer.compute_accuracy(test_loader)}\n```"
                st.markdown(test_acc)

else:
    st.write("Please select a model first.")
