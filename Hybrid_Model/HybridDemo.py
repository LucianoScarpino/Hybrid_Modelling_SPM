from pathlib import Path

import torch

from DatasetLoader import Loader
from Processor import Processing
from Tester import Testing

DATASET_FOLDER = Path(__file__).resolve().parent / "Dataset"
RESULTS_FOLDER = Path(__file__).resolve().parent / "Results"

#data loading
dataloader = Loader()
simplified_dataset = dataloader.get_simplified_dataset(DATASET_FOLDER)
reference_dataset = dataloader.get_reference_dataset(DATASET_FOLDER)
(X_train,y_train),(X_val,y_val),(X_test,y_test) =  dataloader.split_dataset(reference_dataset,simplified_dataset)

train_loader = dataloader.loader(X_train, y_train, shuffle=True)
val_loader = dataloader.loader(X_val, y_val)
test_loader = dataloader.loader(X_test, y_test)

device = dataloader.get_device()

#training parameters
lr = 1e-3
epochs = 10000
early_stopping = 800

#network parameters
input_dim = 3
hidden_dim = 32
output_dim = 101

#training
torch.manual_seed(26)
inventory_offset = Processing.estimate_inventory_offset(
    y_train,
    dataloader.radius
)

print('-'*100)
print(f"Estimated inventory offset: {inventory_offset:.4e}")

processor = Processing(
    train_loader,
    val_loader,
    dataloader.radius,
    inventory_offset
)

trained_model, training_metrics = processor.train(
    lr,
    epochs,
    early_stopping_patience= early_stopping,
    input_dim=input_dim,
    out_dim=output_dim,
    hidden_dim=hidden_dim,
    device=device,
    save_model_checkpoint=False
)

processor.checkpoint_path = processor.save_checkpoint(
    trained_model,
    training_metrics["best_val_loss"],
    optimizer_name="Adam",
    lr=lr,
    training_metrics=training_metrics,
    output_name="selected_checkpoint.pth",
    output_folder=RESULTS_FOLDER / "Models"
)

print(f"Inventory offset: {inventory_offset:.4e}")
print(f"Selected checkpoint: {processor.checkpoint_path.resolve()}")
print('-'*100)

tester = Testing(
    test_loader,
    dataloader.radius,
    dataloader.get_simplified_profiles("test"),
    dataloader.get_tau("test"),
    inventory_offset
)
tester.test(
    trained_model,
    device,
    training_time_seconds=processor.training_time_seconds,
    maximum_training_epochs=processor.maximum_training_epochs,
    completed_training_epochs=processor.completed_training_epochs,
    ending_training_epoch=processor.ending_training_epoch,
    early_stopped=processor.early_stopped,
    output_folder=RESULTS_FOLDER
)

simplified_concentrations, correction = tester.compute_corrections(
    trained_model,
    device,
    simplified_dataset
)

corrected_concentrations = simplified_concentrations + correction

tester.generate_new_dataset(
    simplified_dataset,
    corrected_concentrations,
    DATASET_FOLDER
)
