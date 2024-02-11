import torch

from models.model import SNN
from utils.data_loader import EventsDataset
from trainer import Trainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = EventsDataset(
    datasets=["URFD", "HAR-UP"],
    max_time=30.0,
    nb_steps=1000,
    batch_size=7,
    device=device,
)
x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=0.25)

# dataset.visualize_samples(2)

model = SNN(
    nb_inputs=dataset.nb_pixels,
    nb_hidden=2000,
    nb_outputs=2,
    batch_size=7,
    max_time=dataset.max_time,
    nb_steps=dataset.nb_steps,
    device=device,
)

# Load the model
# model = torch.load("./models/saved/model_v3.pth").to(device)
# model.eval()

# Train the model
trainer = Trainer(model=model)
trainer.train(x_train, y_train, nb_epochs=5, lr=2e-4)

# Save the model
torch.save(model, "./models/saved/model_v4.pth")

# Evaluate the model
print("Training accuracy:", trainer.compute_accuracy(x_train, y_train))
print("Test accuracy:", trainer.compute_accuracy(x_test, y_test))

# x, y = dataset.get_samples()
# print("Validation accuracy:", trainer.compute_accuracy(x, y))

# Visualize the output spikes
trainer.visualize_output(x_train, y_train, 2)
