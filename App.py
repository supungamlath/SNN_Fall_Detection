import configparser
import os
import streamlit as st

# Load the config file
config = configparser.ConfigParser()
config.read("config.txt")

os.environ["root_folder"] = config.get("DEFAULT", "root_dir")

st.set_page_config(
    page_title="Fall Detection SNN",
    page_icon="🧊",
)

st.write("# Welcome to FallSNN Dashboard")

st.markdown(
    """
    This interactive dashboard is designed to demonstrate the capabilities of Spiking Neural Networks (SNNs) in detecting falls, a critical public health issue that predominantly affects older adults. According to the World Health Organization, falls are a leading cause of fatal injuries among the elderly, with an estimated 684,000 fatalities annually. Prompt detection and response to falls are crucial for preventing severe injuries and improving outcomes.

    This approach to fall detection leverages the unique properties of Spiking Neural Networks. Unlike traditional computer vision techniques that rely on handcrafted features and are often limited by environmental variables such as lighting and occlusions, SNNs offer a dynamic and robust solution. They mimic the human brain's processing method, using discrete spikes to encode information and efficiently handle temporal data, making them particularly suitable for real-time fall detection in varied settings.

    ## Features of the Dashboard:
    - **Inspect Datasets:** Explore the datasets used for training the fall detection models.
    - **Create or Load Model:** Start from scratch with a new model and parameters or load a pre-trained model.
    - **Train Model:** Configure and run training sessions with real-time updates on progress.
    - **Evaluate Predictions:** Test the model's predictions using on video data and observe how well the model detects falls.
    - **Analyse Metrics:** Review detailed metrics such as accuracy, and loss to evaluate the effectiveness of the model.

    """
)
