from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


class Loader(object):
    """Load, align, and split datasets for concentration residual learning.

    Each time sample is represented by simplified integral/boundary states,
    while the target is the complete radial FVM--simplified discrepancy.
    """

    def __init__(self, random_state=26):
        self.random_state = random_state

        self.radius = None
        self.tau = None
        self.simplified_profiles = None
        self.reference_profiles = None
        self.split_indices = None

    def get_simplified_dataset(self,data_folder,variant=False,number=None):
        if variant:
            data_folder = data_folder / "Variants"
            name = "simplified_simulation_dataset" + f"_{number}.csv"
            dataset = pd.read_csv(
                data_folder / name
            )
        else:
            dataset = pd.read_csv(
                data_folder / "simplified_simulation_dataset.csv"
            )

        return dataset

    def get_reference_dataset(self,data_folder):
        dataset = pd.read_csv(
            data_folder / "full_simulation_dataset.csv"
        )

        return dataset

    def get_full_dataset(self,data_folder):
        """Compatibility alias for the previous method name."""

        return self.get_simplified_dataset(data_folder)

    @staticmethod
    def _concentration_profiles(dataset):
        """Return a time-by-radius concentration matrix."""

        return (
            dataset
            .pivot(index="time", columns="rho", values="concentration")
            .sort_index()
            .sort_index(axis=1)
        )

    def split_dataset(self, reference_dataset,simplified_dataset):
        """Align reference and simplified grids and create residual splits.

        Returns train, validation, and test tensor pairs ``(features, profile)``.
        Split indices and aligned profiles are retained for later evaluation.
        """

        simplified = self._concentration_profiles(simplified_dataset)
        reference = self._concentration_profiles(reference_dataset)

        simplified_time = simplified.index.to_numpy(dtype=float)
        reference_time = reference.index.to_numpy(dtype=float)
        simplified_radius = simplified.columns.to_numpy(dtype=float)
        reference_radius = reference.columns.to_numpy(dtype=float)

        if not np.allclose(reference_time, simplified_time):
            raise ValueError("Reference and simplified time grids do not match.")

        if not np.allclose(reference_radius, simplified_radius):
            raise ValueError("Reference and simplified radial grids do not match.")

        simplified_states = (
            simplified_dataset
            .drop_duplicates("time")
            .set_index("time")
            .sort_index()
            .loc[simplified.index]
        )

        reference_states = (
            reference_dataset
            .drop_duplicates("time")
            .set_index("time")
            .sort_index()
            .loc[reference.index]
        )

        simplified_current = simplified_states["current"].to_numpy(dtype=float)
        reference_current = reference_states["current"].to_numpy(dtype=float)

        if not np.allclose(reference_current, simplified_current):
            raise ValueError(
                "Reference and simplified current profiles do not match."
            )

        X = simplified_states[[
            "average_concentration",
            "surface_concentration",
            "current",
        ]].to_numpy(dtype=np.float32)

        simplified_profiles = simplified.to_numpy(dtype=np.float32)
        reference_profiles = reference.to_numpy(dtype=np.float32)
        y = reference_profiles - simplified_profiles

        generator = np.random.default_rng(self.random_state)
        shuffled_indices = generator.permutation(len(X))

        number_temporary = int(np.ceil(0.30 * len(X)))
        number_test = int(np.ceil(0.50 * number_temporary))

        train_indices = shuffled_indices[:-number_temporary]
        temporary_indices = shuffled_indices[-number_temporary:]
        validation_indices = temporary_indices[:-number_test]
        test_indices = temporary_indices[-number_test:]

        self.radius = simplified_radius
        self.tau = simplified_states["tau"].to_numpy(dtype=float)
        self.simplified_profiles = simplified_profiles
        self.reference_profiles = reference_profiles
        self.split_indices = {
            "train": train_indices,
            "validation": validation_indices,
            "test": test_indices,
        }

        def tensors(selected_indices):
            return (
                torch.from_numpy(X[selected_indices]),
                torch.from_numpy(y[selected_indices]),
            )

        return (
            tensors(train_indices),
            tensors(validation_indices),
            tensors(test_indices),
        )

    def loader(self, X, y, batch_size=32, shuffle=False):
        """Wrap feature and target tensors in a configured ``DataLoader``."""
        dataset = TensorDataset(X, y)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def get_simplified_profiles(self, split):
        """Return simplified profiles associated with a previously built split."""
        if self.split_indices is None:
            raise RuntimeError("split_dataset must be called first.")

        if split not in self.split_indices:
            raise ValueError(
                "split must be 'train', 'validation', or 'test'."
            )

        indices = self.split_indices[split]

        return torch.from_numpy(self.simplified_profiles[indices])

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
