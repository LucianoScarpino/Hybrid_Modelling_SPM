from pathlib import Path

import torch

from Visualizer import HybridSimulationVisualizer


PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = PROJECT_FOLDER / "Dataset"
IMAGES_FOLDER = PROJECT_FOLDER / "Images"
MODELS_FOLDER = PROJECT_FOLDER / "Results" / "Models"

CORRECTED_DATASET = DATASET_FOLDER / "corrected_dataset.csv"
SIMPLIFIED_DATASET = DATASET_FOLDER / "simplified_simulation_dataset.csv"
REFERENCE_DATASET = DATASET_FOLDER / "full_simulation_dataset.csv"

SAVE_IMAGES = True
SHOW_IMAGES = True


def newest_selected_checkpoint(models_folder):
    """Return the most recently saved selected checkpoint."""
    checkpoint_paths = list(
        models_folder.glob("selected_checkpoint_*.pth")
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            "No selected checkpoint was found. Run HybridDemo.py first."
        )

    return max(
        checkpoint_paths,
        key=lambda checkpoint_path: checkpoint_path.stat().st_mtime
    )

if not CORRECTED_DATASET.exists():
    raise FileNotFoundError(
        "The corrected dataset does not exist. Run HybridDemo.py first."
    )

if not REFERENCE_DATASET.exists():
    raise FileNotFoundError(
        "The reference dataset does not exist."
    )

if not SIMPLIFIED_DATASET.exists():
    raise FileNotFoundError(
        "The simplified dataset does not exist."
    )

checkpoint_path = newest_selected_checkpoint(MODELS_FOLDER)
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

if checkpoint.get("constraint") != "constant_volume_average":
    raise ValueError(
        "The selected checkpoint does not use the constant-volume-average "
        "constraint. Run HybridDemo.py again."
    )

visualizer = HybridSimulationVisualizer(
    corrected_dataset=CORRECTED_DATASET,
    simplified_dataset=SIMPLIFIED_DATASET,
    reference_dataset=REFERENCE_DATASET,
    inventory_offset=float(checkpoint["inventory_offset"]),
    images_folder=IMAGES_FOLDER,
    compare=False,
    save_im=SAVE_IMAGES,
    show_im=SHOW_IMAGES
)

visualizer.show_all()
