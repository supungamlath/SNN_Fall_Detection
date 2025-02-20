from datetime import datetime
import os
import streamlit as st
from utils.SpikingDataLoader import SpikingDataLoader
from utils.SpikingDataset import SpikingDataset
from utils.BinaryTrainer import BinaryTrainer
from utils.streamlit_helpers import print_loss_accuracy
from utils.visualization import live_plot_plotly
from utils.helpers import get_datasets_dirs, load_params, save_params

datasets_dirs = get_datasets_dirs()

st.set_page_config(page_title="Fall Detection SNN", page_icon="🧊", layout="wide")

st.write("# Training")

# Define the directory where the previous training runs are stored
training_records_file = f"{os.environ['root_folder']}/models/saved/training_runs.json"

# Load previous training runs
training_records = load_params(training_records_file)

model_name = st.session_state.get("model_name", None)
st.write(f"### Selected Model: {model_name}")

if model_name:
    with st.expander("Previous Training Runs"):
        if model_name in training_records:
            previous_training_runs = training_records[model_name]
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
                if "train_metrics_hist" in run:
                    live_plot_plotly(
                        [epoch["accuracy"] for epoch in run["train_metrics_hist"]],
                        [epoch["accuracy"] for epoch in run["dev_metrics_hist"]],
                        title="Model Accuracy",
                        y_label="Accuracy",
                        renderer=st.plotly_chart,
                    )
                else:
                    st.write("`No training history available.`")

                if st.button("Delete Training Run", key=run["datetime"]):
                    previous_training_runs.remove(run)
                    save_params(training_records_file, training_records)
                st.divider()
        else:
            st.write("No training runs yet.")

    st.write("New Training Run")
    start_training_disabled = False
    selected_dataset = st.selectbox("Dataset", list(datasets_dirs.keys()))
    model_params = st.session_state["model_params"]
    dataset = SpikingDataset(
        root_dir=datasets_dirs[selected_dataset],
        time_duration=model_params["time_duration"],
    )

    train_test_ratio = st.slider("Train/Test Ratio", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
    nb_test_samples = int(len(dataset) * train_test_ratio)
    nb_test_samples = (nb_test_samples // model_params["batch_size"]) * model_params["batch_size"]
    nb_train_samples = len(dataset) - nb_test_samples
    nb_train_samples = (nb_train_samples // model_params["batch_size"]) * model_params["batch_size"]
    cols = st.columns(6)
    with cols[2]:
        st.write(f"Train Samples: {nb_train_samples}")
    with cols[3]:
        st.write(f"Test Samples: {nb_test_samples}")

    nb_epochs = st.number_input("Number of Epochs", min_value=1, max_value=500, value=20)
    early_stopping_option = st.checkbox("Enable early stopping", value=True)
    learning_rate = st.number_input("Initial learning rate", min_value=1e-5, max_value=1e-1, value=2e-4, format="%.5f")

    if st.button("Start Training Run", type="primary", disabled=start_training_disabled):

        if model_name not in training_records:
            training_records[model_name] = []
        training_records[model_name].append(
            {
                "datetime": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "dataset": selected_dataset,
                "train_test_ratio": train_test_ratio,
                "nb_epochs": nb_epochs,
                "learning_rate": learning_rate,
            }
        )
        save_params(training_records_file, training_records)

        with st.spinner("Loading datasets..."):
            train_dataset, test_dataset = dataset.random_split(test_size=train_test_ratio, shuffle=True)
            train_loader = SpikingDataLoader(
                train_dataset, batch_size=model_params["batch_size"], nb_steps=model_params["nb_steps"], shuffle=False
            )
            test_loader = SpikingDataLoader(
                test_dataset, batch_size=model_params["batch_size"], nb_steps=model_params["nb_steps"], shuffle=False
            )

        with st.spinner("Training the model..."):
            model = st.session_state["model"]
            trainer = BinaryTrainer(model=model)

            train_metrics_hist, dev_metrics_hist = [], []

            def evaluate_callback(metrics, epoch):
                dev_metrics_hist.append(metrics)

            def train_callback(metrics, epoch):
                train_metrics_hist.append(metrics)

            trainer.train(
                train_loader,
                nb_epochs=nb_epochs,
                lr=learning_rate,
                evaluate_dataloader=test_loader,
                evaluate_callback=evaluate_callback,
                train_callback=train_callback,
                stop_early=early_stopping_option,
            )
            training_records[model_name][-1]["train_metrics_hist"] = train_metrics_hist
            training_records[model_name][-1]["dev_metrics_hist"] = dev_metrics_hist
            save_params(training_records_file, training_records)

        with st.spinner("Saving the model..."):
            model.save(f"{os.environ['root_folder']}/models/saved/{model_name}.pth")

        col1, col2 = st.columns(2)
        with col1:
            live_plot_plotly(
                [epoch["loss"] for epoch in train_metrics_hist],
                title="Training Loss",
                y_label="Loss",
                renderer=st.plotly_chart,
            )
        with col2:
            live_plot_plotly(
                [epoch["accuracy"] for epoch in train_metrics_hist],
                [epoch["accuracy"] for epoch in dev_metrics_hist],
                title="Model Accuracy",
                y_label="Accuracy",
                renderer=st.plotly_chart,
            )
        st.success("Training run completed successfully.")

else:
    st.write("Please select a model first.")
