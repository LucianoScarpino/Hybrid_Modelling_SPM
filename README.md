# Hybrid Modelling for Lithium Diffusion in a Spherical Particle

This repository investigates physics-based, physics-informed, and hybrid
machine-learning approaches for lithium diffusion inside a spherical active
material particle. The project uses a thermo-diffusive Single Particle Model
(SPM) as a high-fidelity reference and studies how simplified physics can be
combined with data-driven corrections.

The software covers four connected modelling tasks:

- a conservative finite-volume reference model with state-dependent solid
  diffusivity and lumped thermal dynamics;
- a Physics-Informed Neural Network (PINN) constrained by a simplified,
  constant-diffusivity equation;
- post-training residual correction of a frozen PINN through a feedforward
  neural network (Hybrid-PINN);
- residual correction of a simplified finite-volume model and identification
  of the thermal dynamics using either SINDYc or a feedforward neural network.

The main objective is not only to reduce prediction error, but also to compare
accuracy, physical consistency, computational cost, generalization, and model
interpretability across the different approaches.

> **Full documentation:** [open the complete technical report](./Hybrid_PINN_for_SPM.pdf).

## Modelling workflow

```mermaid
flowchart LR
    A["Full thermo-diffusive SPM"] --> B["Finite-volume reference datasets"]
    C["Simplified diffusion physics"] --> D["Standalone PINN"]
    B --> D
    D --> E["Residual FFN"]
    E --> F["Hybrid-PINN"]
    C --> G["Simplified FVM baseline"]
    B --> H["Radial-profile residual FFN"]
    G --> H
    H --> I["FVM-FFN hybrid model"]
    B --> J["Thermal trajectories"]
    J --> K["SINDYc"]
    J --> L["Thermal FFN"]
```

The two hybrid concentration models serve different purposes:

- **Hybrid-PINN:** corrects the output of an already trained and frozen PINN at
  each space--time coordinate.
- **FVM--FFN hybrid:** predicts an entire radial discrepancy profile from
  simplified-model state quantities such as average concentration, surface
  concentration, and applied current.

## Repository structure

```text
Software/
├── Hybrid_PINN_for_SPM.pdf          # Complete technical report
├── Single_Particle_Model_V3/        # Full and simplified FVM particle models
│   ├── Config.py                    # Command-line physical/numerical settings
│   ├── ModelGeneration.py           # Geometry, diffusivity, thermal law, flux
│   ├── Assembler.py                 # Conservative finite-volume equations
│   ├── ParticleSimulation.py        # Coupled simulation orchestration
│   ├── DatasetGenerator.py          # Checks, derived quantities, CSV export
│   ├── Visualizer.py                # Concentration/temperature diagnostics
│   ├── Demo.py                      # Main simulation and dataset entry point
│   ├── ConvergenceTest.py           # Radial mesh-convergence study
│   ├── SPMDataset/                  # Generated reference/training datasets
│   └── Images/                      # FVM figures
│
├── PINN/                            # Simplified-physics PINN and Hybrid-PINN
│   ├── NeuralNetwork.py             # PINN, residual FFN, hybrid wrapper
│   ├── Generator.py                 # Data/collocation/boundary/residual loaders
│   ├── Processor.py                 # Adam and L-BFGS PINN training
│   ├── Tester.py                    # Held-out error and physics evaluation
│   ├── Visualizer.py                # Standalone PINN plots and comparisons
│   ├── PINNDemo.py                  # Standalone PINN training entry point
│   ├── ResidualLearning.py          # Residual training and physical checks
│   ├── HybridDemo.py                # Frozen-PINN correction pipeline
│   ├── HybridVisualizer.py          # Hybrid-PINN diagnostic plots
│   ├── PlotPINN.py                  # Plot from saved checkpoints
│   ├── Data/                        # PINN and corrected datasets
│   ├── Results/                     # Checkpoints and metric histories
│   └── Images/                      # PINN/Hybrid-PINN figures
│
└── Hybrid_Model/                    # FVM--FFN and thermal hybrid studies
    ├── DatasetLoader.py             # Aligned profile datasets and splits
    ├── NeuralNetwork.py             # Profile-correction and thermal FFNs
    ├── Processor.py                 # Profile-correction training
    ├── Tester.py                    # Reconstruction, metrics, corrected CSV
    ├── HybridDemo.py                # FVM--FFN hybrid training entry point
    ├── Visualizer.py                # Corrected-field visualization
    ├── PlotHybrid.py                # Plot a saved corrected dataset
    ├── ThermalDiscover.py           # SINDYc/FFN identification and rollout
    ├── TemperatureField.py          # Thermal model selection and comparison
    ├── CompareMethods.py            # Cross-method concentration comparison
    ├── ConvergenceTest.py           # Robustness across dataset variants
    ├── Dataset/                     # Baseline, reference, corrected, variants
    ├── Results/                     # Training/test histories and checkpoints
    └── Images/                      # Hybrid and thermal figures
```

Generated datasets, model checkpoints, metric histories, and figures are kept
next to the workflow that produces them. Most entry-point scripts therefore
expect to be launched from their own subdirectory.

## Representative results

### Full-physics finite-volume reference

The reference solver resolves the radial concentration field while coupling
solid diffusion to the lumped temperature state and the operating-current
profile.

![Finite-volume concentration field](Single_Particle_Model_V3/Images/concentration_map.png)

### Post-training correction of the PINN

The frozen PINN supplies the physics-informed baseline. The residual network
learns the discrepancy with respect to the full FVM field, substantially
reducing the error without retraining the original PINN. Physical properties
not enforced by the residual architecture are evaluated separately through
mass-balance, bounds, boundary-flux, PDE, and regularity checks.

![FVM, PINN, and Hybrid-PINN field comparison](PINN/Images/Comparison/PINN_Hybrid_concentration_fields_comparison.png)

### Thermal model identification

SINDYc and the thermal FFN are trained on multiple operating trajectories and
tested on dedicated interpolation and extrapolation cases. Their comparison
includes temperature and derivative errors, rollout stability, training and
inference time, and model complexity.

![SINDYc and FFN interpolation comparison](Hybrid_Model/Images/Thermal_models_comparison_interpolation.png)

## Typical execution order

The scripts currently use local relative paths, so enter the corresponding
folder before running them.

```bash
# 1. Generate/reference the finite-volume simulations and datasets
cd Single_Particle_Model_V3
python Demo.py

# 2. Train and evaluate the standalone PINN
cd ../PINN
python PINNDemo.py

# 3. Train the post-processing residual correction of the frozen PINN
python HybridDemo.py

# 4. Regenerate PINN or Hybrid-PINN figures from saved checkpoints
python PlotPINN.py --mode all

# 5. Train the simplified-FVM radial-profile correction
cd ../Hybrid_Model
python HybridDemo.py

# 6. Compare SINDYc and the thermal FFN
python TemperatureField.py
```

The main experiment controls are currently defined in `Config.py` or near the
top of the relevant demo script. In particular, `TemperatureField.py` exposes
manual controls for the selected model (`SINDYC`, `FFN`, or `COMPARISON`) and
for interpolation versus extrapolation testing.

## Main Python dependencies

The implementation uses Python 3.11 with:

- NumPy, pandas, SciPy, and scikit-learn;
- PyTorch;
- Matplotlib;
- PySINDy.

Dataset generation and full training can be computationally expensive. Saved
datasets and checkpoints allow plotting and post-processing to be repeated
without retraining every model.

## Report

The mathematical formulation, numerical discretization, training procedures,
physical-consistency diagnostics, and complete quantitative comparisons are
available in the [full project report](./Hybrid_PINN_for_SPM.pdf).
