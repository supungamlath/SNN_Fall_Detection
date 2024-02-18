import torch
from data.SpikingDataLoader import SpikingDataLoader
from data.SpikingDataset import SpikingDataset

from models.model import SNN

from utils.Trainer import Trainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = SpikingDataset(
    root_dir="./data/urfd-spiking-dataset-240", max_time=15.0, nb_steps=1000
)

# Splitting the dataset
train_dataset, test_dataset = dataset.random_split(test_size=0.25, shuffle=True)

model = SNN(
    nb_inputs=dataset.nb_pixels,
    nb_hidden=2000,
    nb_outputs=2,
    batch_size=7,
    max_time=dataset.max_time,
    nb_steps=dataset.nb_steps,
    device=device,
)

# Creating DataLoader instances
train_loader = SpikingDataLoader(train_dataset, batch_size=7, shuffle=True)
test_loader = SpikingDataLoader(test_dataset, batch_size=7, shuffle=False)

# Load the model
model = torch.load("./models/saved/model_v5.pth").to(device)
# model.eval()

# Train the model
trainer = Trainer(model=model)
# trainer.train(train_loader, nb_epochs=5, lr=2e-4)

# Save the model
# torch.save(model, "./models/saved/model_v7.pth")

# Evaluate the model
# print("Training accuracy:", trainer.compute_accuracy(train_loader))
# print("Test accuracy:", trainer.compute_accuracy(test_loader))

# x, y = dataset.get_samples()
# print("Validation accuracy:", trainer.compute_accuracy(x, y))

# Visualize the output spikes
trainer.visualize_output(test_loader, 1)
