import torch
import pandas as pd

from Generator import DatasetGenerator
from Processor import Processing
from Tester import Testing
from ResidualLearning import ResidualLearner

#define device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#constants definition
initial_concentration = 0.5
delta_0 = -0.20046160073059885          #corresponding to dimensionless flux with D(C,T) = D_ref
tau_final = 1.0073356401384084

lambda_data = 1.0
lambda_pde = 1.0
lambda_bc = 1.0

number_rho_points = 201
number_tau_points = 201

#datasets reading and generation
solver_data = "./Data/pinn_training_dataset.csv"
generator = DatasetGenerator(solver_data,length=20301)

generator.generate_collocation_points()
generator.generate_boundary_points(tauf=tau_final)

collocation_data = "./Data/collocation_dataset.csv"
boundary_data = "./Data/boundary_dataset.csv"

data_points = pd.read_csv(solver_data).values
collocation_points = pd.read_csv(collocation_data).values
boundary_points = pd.read_csv(boundary_data).values

#data
d_train_dataset, d_val_dataset, d_test_dataset = generator.generate_dataloaders(data_points,labels=True,batch_size=512)             #(rho,tau),C
f_train_dataset, f_val_dataset, f_test_dataset = generator.generate_dataloaders(collocation_points,labels=False,batch_size=512)     #(rho,tau)
b_train_dataset, b_val_dataset, b_test_dataset = generator.generate_dataloaders(boundary_points,labels=False,batch_size=512)        #(1,tau))

#training
epochs = 10000
lr = 1e-3
scheduler_patience = 200
early_stopping_patience = 800
lbfgs_max_iter = 500

print("=" * 100)
print("PINN DEMO")
print(f"Device: {device}")
print(
    "Configuration | "
    f"Epochs: {epochs} | "
    f"Learning rate: {lr:.2e} | "
    f"Scheduler patience: {scheduler_patience} | "
    f"Early-stopping patience: {early_stopping_patience} | "
    f"LBFGS max iterations: {lbfgs_max_iter}"
    )
print(
    "Loss weights | "
    f"Data: {lambda_data:.3g} | "
    f"PDE: {lambda_pde:.3g} | "
    f"BC: {lambda_bc:.3g}"
    )
print(
    "Training samples | "
    f"Data: {len(d_train_dataset.dataset)} | "
    f"Collocation: {len(f_train_dataset.dataset)} | "
    f"Boundary: {len(b_train_dataset.dataset)}"
    )
print(
    "Validation samples | "
    f"Data: {len(d_val_dataset.dataset)} | "
    f"Collocation: {len(f_val_dataset.dataset)} | "
    f"Boundary: {len(b_val_dataset.dataset)}"
    )
print(
    "Test samples | "
    f"Data: {len(d_test_dataset.dataset)} | "
    f"Collocation: {len(f_test_dataset.dataset)} | "
    f"Boundary: {len(b_test_dataset.dataset)}"
    )
print("=" * 100)
print("Starting training...")

processor = Processing(
    d_train_dataset,
    f_train_dataset,
    b_train_dataset,
    d_val_dataset,
    f_val_dataset,
    b_val_dataset,
    device
    )

solver = processor.train(
    epochs= epochs,
    init_concentration=initial_concentration,
    lambda_d=lambda_data,
    lambda_f= lambda_pde,
    lambda_b=lambda_bc,
    flux=delta_0,
    lr=lr,
    early_stopping_patience=early_stopping_patience,
    sheduler_patience=scheduler_patience,
    lbfgs_max_iter=lbfgs_max_iter
    )

print("Training completed.")
print("-" * 100)
print("Starting test evaluation...")

tester = Testing(
    d_test_dataset,
    f_test_dataset,
    b_test_dataset
)

metrics = tester.test(
    solver,
    initial_concentration,
    delta_0,
    )

print("Test evaluation completed.")
print("=" * 100)

tester.save_test_results(
    metrics,
    device,
    initial_concentration,
    delta_0,
    output_folder="./Results",
    filename="test_history.csv"
)

print(f"Test saved in Results folder -> test_history.csv")
print("=" * 100)
print("TEST RESULTS")
print(
    "Concentration | "
    f"Relative L2 error: "
    f"{metrics['relative_concentration_error']:.4e} "
    f"({100.0 * metrics['relative_concentration_error']:.4f}%)"
    )
print(
    "Surface concentration | "
    f"Relative L2 error: "
    f"{metrics['relative_surface_error']:.4e} "
    f"({100.0 * metrics['relative_surface_error']:.4f}%)"
    )
print(
    "Physics residuals | "
    f"PDE RMSE: {metrics['pde_rmse']:.4e} | "
    f"BC RMSE: {metrics['boundary_rmse']:.4e}"
    )
print(
    "Inference | "
    f"Total: {1e3 * metrics['inference_time_seconds']:.4f} ms | "
    f"Per sample: "
    f"{1e6 * metrics['inference_time_seconds_per_sample']:.4f} us"
    )
print("=" * 100)