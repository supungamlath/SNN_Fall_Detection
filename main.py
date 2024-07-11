import os

from models.SpikingNN import SpikingNN
from utils.SpikingDataset import SpikingDataset
from utils.SpikingDataLoader import SpikingDataLoader

from utils.Trainer import Trainer

if os.name == "nt":
    os.environ["root_folder"] = "E:/Projects/PythonProjects/SNN"
else:
    os.environ["root_folder"] = "/content/SNN_Fall_Detection"

dataset = SpikingDataset(
    root_dir=f"{os.environ['root_folder']}/data/urfd-spiking-dataset-240",
    max_time=15.0,
    nb_steps=1000,
)

# Splitting the dataset
train_dataset, test_dataset = dataset.random_split(test_size=0.2, shuffle=True)

model = SpikingNN(
    layer_sizes=[dataset.nb_pixels, 1000, 50, 2],
    nb_steps=dataset.nb_steps,
)

# Creating DataLoader instances
train_loader = SpikingDataLoader(train_dataset, batch_size=7, shuffle=True)
test_loader = SpikingDataLoader(test_dataset, batch_size=7, shuffle=False)

# Load the model
# model = SpikingNN.load(f"{os.environ['root_folder']}/models/saved/model_v5.pth")
# model.eval()

# Train the model
trainer = Trainer(model=model)
trainer.train(train_loader, nb_epochs=5, lr=2e-4)

# Save the model
model.save(f"{os.environ['root_folder']}/models/saved/model_v4.pth")

# Visualize the output spikes
trainer.visualize_output(test_loader, 1)
