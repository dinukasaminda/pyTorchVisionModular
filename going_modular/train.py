"""
  Trains a PyTorch image classification model using device-agnostic code.

"""
print("Starting train script...")

import os
import argparse
from timeit import default_timer as timer

import torch
from torchvision import transforms

import data_setup, engine, model_builder, utils

# Initialize the parser
parser = argparse.ArgumentParser(description="Tiny VGG Model for food classification.")

# Add arguments
parser.add_argument("--num_epochs", type=int, help="Number of epochs")
parser.add_argument("--batch_size", type=int, help="Size of the batch")
parser.add_argument("--lr", type=float, help="Learning rate")

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
NUM_WORKERS = os.cpu_count()
print(f"[INFO] Number of cpu: {NUM_WORKERS}")

# Parse the arguments
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS=args.num_epochs
BATCH_SIZE=args.batch_size
HIDDEN_UNITS=10
LEARNING_RATE=args.lr

# Setup directories
train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'
model_save_dir='models'
model_name="05_going_modular_cell_mode_tinyvgg_model.pth"

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# Create transorms
# This will also normalize color values (0-255) to 0 - 1
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Create dataloaders and get claa_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
                                                    train_dir=train_dir,
                                                    test_dir=test_dir,
                                                    transform=data_transform,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=NUM_WORKERS
                                                )

# Create model
model = model_builder.TinyVGG(input_shape=3, # number of color channels (3 for RGB)
                    hidden_units=HIDDEN_UNITS,
                    output_shape=len(class_names)).to(device)

# Setup loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Start training with help from engine.py
model_results = engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device)


# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
print("[INFO] Training done.")

# Save the model
utils.save_model(model=model,
            target_dir=model_save_dir,
            model_name=model_name)

print("[INFO] Done train script execution.")
