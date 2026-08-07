from pathlib import Path

from Visualizer import HybridSimulationVisualizer


PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = PROJECT_FOLDER / "Dataset"
IMAGES_FOLDER = PROJECT_FOLDER / "Images"

CORRECTED_DATASET = DATASET_FOLDER / "corrected_dataset.csv"
REFERENCE_DATASET = DATASET_FOLDER / "full_simulation_dataset.csv"

SAVE_IMAGES = True
SHOW_IMAGES = True

if not CORRECTED_DATASET.exists():
    raise FileNotFoundError(
        "The corrected dataset does not exist. Run HybridDemo.py first."
    )

if not REFERENCE_DATASET.exists():
    raise FileNotFoundError(
        "The reference dataset does not exist."
    )

visualizer = HybridSimulationVisualizer(
    corrected_dataset=CORRECTED_DATASET,
    reference_dataset=REFERENCE_DATASET,
    images_folder=IMAGES_FOLDER,
    save_im=SAVE_IMAGES,
    show_im=SHOW_IMAGES
)

visualizer.show_all()