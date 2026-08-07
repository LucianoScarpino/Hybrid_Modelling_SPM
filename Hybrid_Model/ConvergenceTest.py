from pathlib import Path

import torch

from DatasetLoader import Loader
from NeuralNetwork import FNN
from Tester import Testing


PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = PROJECT_FOLDER / "Dataset"
VARIANTS_FOLDER = DATASET_FOLDER / "Variants"
RESULTS_FOLDER = PROJECT_FOLDER / "Results"

CHECKPOINT_PATH = (RESULTS_FOLDER
    / "Models"
    / "checkpoint_20260803_155542_385714.pth"
)

number = 4

#data loading
dataloader = Loader()

simplified_dataset = dataloader.get_simplified_dataset(DATASET_FOLDER,variant=True,number=number)
reference_dataset = dataloader.get_reference_dataset(DATASET_FOLDER)

(X_train,y_train),(X_val,y_val),(X_test,y_test) = (dataloader.split_dataset(reference_dataset,simplified_dataset))

#use the complete variant as test dataset
X_test = torch.cat((X_train,X_val,X_test),dim=0)
y_test = torch.cat((y_train,y_val,y_test),dim=0)

simplified_profiles = torch.cat(
    (
        dataloader.get_simplified_profiles("train"),
        dataloader.get_simplified_profiles("validation"),
        dataloader.get_simplified_profiles("test")
    ),
    dim=0
)

test_loader = dataloader.loader(X_test,y_test)
device = dataloader.get_device()

#load the frozen nominal network
trained_model = FNN(
    input_dim=3,
    hidden_dim=32,
    output_dim=101
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

trained_model.load_state_dict(
    checkpoint["model_state_dict"]
)

trained_model.eval()

#testing
tester = Testing(
    test_loader,
    dataloader.radius,
    simplified_profiles
)

tester.test(
    trained_model,
    device,
    training_time_seconds=0.0,
    maximum_training_epochs=0,
    completed_training_epochs=0,
    ending_training_epoch=0,
    early_stopped=False,
    output_folder=RESULTS_FOLDER,
    filename="test_history_variants.csv",
    variant_number=number
)

#generate corrected variant
simplified_concentrations,correction = (
    tester.compute_corrections(
        trained_model,
        device,
        simplified_dataset
    )
)

corrected_concentrations = (
    simplified_concentrations
    + correction
)

tester.generate_new_dataset(
    simplified_dataset,
    corrected_concentrations,
    VARIANTS_FOLDER,
    filename=f"corrected_dataset_{number}.csv"
)
