import os
import pandas as pd
from collections import Counter


def label_URFD_dataset():
    dataset_folder = f"{os.environ['root_folder']}/data/urfd-spiking-dataset-240"
    folder_names = [
        folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))
    ]
    labels = [1 if "fall" in name else 0 for name in folder_names]
    df = pd.DataFrame({"folder_name": folder_names, "label": labels})
    df.to_csv(f"{dataset_folder}/labels.csv", index=False)


def label_HAR_UP_dataset():
    dataset_folder = f"{os.environ['root_folder']}/data/har-up-spiking-dataset-240"
    folder_names = [
        folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))
    ]
    labels = [1] * len(folder_names)
    df = pd.DataFrame({"folder_name": folder_names, "label": labels})
    df.to_csv(f"{dataset_folder}/labels.csv", index=False)


def create_windowed_labels(data_path, window_size, total_duration=60, padding_label=0.0, replace_with_padding=20.0):
    """
    Create labels CSV with most frequent Tags in specified time windows.

    Args:
        data_path (str): Path to the input CSV file
        window_size (float): Size of the time window in seconds
        total_duration (int): Total duration to process in seconds (default: 60)
        padding_label (int): Label to use for padding incomplete videos
        replace_with_padding (float): Replace unknown data with padding label (default: 20.0)

    Returns:
        pd.DataFrame: DataFrame containing the processed labels
    """
    # Read and preprocess the data
    data = pd.read_csv(data_path)
    data.drop(index=data.index[0], axis=0, inplace=True)

    # Parse the formats of the columns
    data["TimeStamps"] = pd.to_datetime(data["TimeStamps"])
    data["Subject"] = data["Subject"].astype(int)
    data["Activity"] = data["Activity"].astype(int)
    data["Trial"] = data["Trial"].astype(int)

    # Calculate number of windows for the total duration
    num_windows = int(total_duration // window_size)

    # Extract unique combinations of Subject, Activity, and Trial
    grouped = data.groupby(["Subject", "Activity", "Trial"])

    # Process each group
    video_info = []
    for (subject, activity, trial), group in grouped:
        start_time = group["TimeStamps"].min()
        end_time = group["TimeStamps"].max()
        duration = (end_time - start_time).total_seconds()
        video_name = f"Subject{subject}Activity{activity}Trial{trial}"

        # Initialize labels for each window
        window_labels = []

        # Process each time window
        for i in range(num_windows):
            window_start = start_time + pd.Timedelta(seconds=i * window_size)
            window_end = window_start + pd.Timedelta(seconds=window_size)

            # Get Tags in current window
            window_data = group[(group["TimeStamps"] >= window_start) & (group["TimeStamps"] < window_end)]

            if len(window_data) > 0:
                # Get most frequent Tag in the window
                tags = window_data["Tag"].values
                most_common_tag = Counter(tags).most_common(1)[0][0]
                if 1.0 <= most_common_tag and most_common_tag <= 5.0:
                    window_labels.append(1.0)
                else:
                    window_labels.append(0.0)
            else:
                # If no data in window, use padding label
                window_labels.append(padding_label)

        # Replace unknown data with padding label
        window_labels = [padding_label if label == replace_with_padding else label for label in window_labels]

        # Create entries for both cameras
        for camera in [1, 2]:
            entry = {"name": f"{video_name}Camera{camera}", "length": duration}
            # Add window labels
            for i, label in enumerate(window_labels):
                entry[f"window_{i+1}"] = label
            video_info.append(entry)

    # Create DataFrame for result
    result_df = pd.DataFrame(video_info)

    return result_df


if __name__ == "__main__":
    root_folder = os.environ.get("root_folder", "E:/Projects/PythonProjects/SNN")
    input_path = f"{root_folder}/data/har-up-spiking-dataset-240/CompleteDataSet.csv"
    output_path = f"{root_folder}/data/har-up-spiking-dataset-240/labels_binary_w0.1.csv"

    # Create labels with 1-second windows for 60 seconds total duration
    result_df = create_windowed_labels(input_path, window_size=0.1, total_duration=60, padding_label=0.0)

    # Save to CSV
    result_df.to_csv(output_path, index=False)
