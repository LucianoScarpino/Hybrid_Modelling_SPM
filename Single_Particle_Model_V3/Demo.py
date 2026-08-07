import numpy as np

from ParticleSimulation import Simulate
from DatasetGenerator import PostProcessing
from Visualizer import SimulationVisualizer
from Config import parse_params

args = parse_params()
generate_dataset = True             #<--change to whether generate dataset or not

is_reference_model = (args.Ds_type == 'adaptive' and args.profile == 'operating')
is_simplified_model = (args.Ds_type == 'constant' and args.profile == 'operating')
is_variant = args.variant_number is not None


simulation = Simulate(args)
solutions = simulation.run()

tau = solutions["tau"]
time = solutions["time"]
radius = solutions["radius"]
concentration = solutions["concentration"]
current_profile = solutions['currents']
temperature = solutions["temperatures"]

surf_concentration_evolution = concentration[-1,:]

max_index = np.unravel_index(
    np.argmax(concentration),
    concentration.shape
    )

radial_index, time_index = max_index

post = PostProcessing(**solutions)
expected_mass_balance = post.compute_expected_mass_balance()
mass_balance, residual = post.verify_mass_balance()
average_concentration = post.compute_average_concentration()
average_max_index = np.argmax(average_concentration)

physical_bounds = post.verify_physical_bounds()
temperature_passed, temperature_checks = post.verify_temperature()

dataset,global_output = post.generate_dataset()
PINN_X_training, PINN_Y_training, _ = post.get_pinn_training_arrays()

FOLDER_PATH = './SPMDataset/Variants' if is_variant else './SPMDataset'

if generate_dataset:
    if is_reference_model:
        full_dataset_filename = (
            f"full_simulation_dataset_{args.variant_number}.csv"
            if is_variant else "full_simulation_dataset.csv"
        )
        pinn_dataset_filename = (
            f"pinn_training_dataset_{args.variant_number}.csv"
            if is_variant else "pinn_training_dataset.csv"
        )

        post.save_dataset(
                FOLDER_PATH,
                dataset,
                PINN_X_training,
                PINN_Y_training,
                full_dataset_filename,
                pinn_dataset_filename
                )
    elif is_simplified_model:
        simplified_dataset_name = (
            f"simplified_simulation_dataset_{args.variant_number}.csv"
            if is_variant else "simplified_simulation_dataset.csv"
        )

        post.save_simplified_dataset(
            tau,
            time,
            radius,
            concentration,
            average_concentration,
            surf_concentration_evolution,
            current_profile,
            FOLDER_PATH,
            simplified_dataset_name
        )

if not is_variant:
    visualizer = SimulationVisualizer(
        tau=tau,
        time=time,
        radius=radius,
        concentration=concentration,
        save_im=True                       #<-- change to whether save images or not
    )

    visualizer.show_all(
        average_concentration,
        residual,
        concentration,
        tau,
        args.profile,
        temperature
    )

print("\n==================== SIMULATION SUMMARY ====================")

print("\n[ARRAY DIMENSIONS]")
print(f"Dimensionless time vector (tau): {tau.shape}")
print(f"Physical time vector (t):        {time.shape}")
print(f"Radial grid (rho):              {radius.shape}")
print(f"Concentration field:            {concentration.shape}")

print("\n[TIME DOMAIN]")
print(f"Initial dimensionless time : {tau[0]:.6f}")
print(f"Final dimensionless time   : {tau[-1]:.6f}")
print(f"Initial physical time [s]  : {time[0]:.3f}")
print(f"Final physical time [s]    : {time[-1]:.3f}")

print("\n[RADIAL DOMAIN]")
print(f"Particle centre (rho)      : {radius[0]:.3f}")
print(f"Particle surface (rho)     : {radius[-1]:.3f}")
print(f"Radial spacing             : {radius[1]-radius[0]:.5f}")

print("\n[INITIAL CONCENTRATION]")
print(f"Centre concentration       : {concentration[0,0]:.6f}")
print(f"Surface concentration      : {concentration[-1,0]:.6f}")
print(f"Minimum concentration      : {concentration[:,0].min():.6f}")
print(f"Maximum concentration      : {concentration[:,0].max():.6f}")

print("\n[FINAL CONCENTRATION]")
print(f"Centre concentration       : {concentration[0,-1]:.6f}")
print(f"Surface concentration      : {concentration[-1,-1]:.6f}")
print(f"Minimum concentration      : {concentration[:,-1].min():.6f}")
print(f"Maximum concentration      : {concentration[:,-1].max():.6f}")

print("\n[CONCENTRATION EVOLUTION]")
print(f"Centre variation           : {concentration[0,-1]-concentration[0,0]:+.6e}")
print(f"Surface variation          : {concentration[-1,-1]-concentration[-1,0]:+.6e}")
print(f"Surface - Centre (final)   : {concentration[-1,-1]-concentration[0,-1]:+.6e}")

print("\n[GLOBAL MAXIMUM LOCATION]")
print(f"Maximum concentration      : {concentration[radial_index, time_index]:.6f}")
print(f"Radial coordinate          : {radius[radial_index]:.6f}")
print(f"Dimensionless time         : {tau[time_index]:.6f}")
print(f"Physical time [s]          : {time[time_index]:.3f}")

print("\n[VOLUME-AVERAGED CONCENTRATION]")
print(f"Initial average concentration : {average_concentration[0]:.6f}")
print(f"Maximum average concentration : {average_concentration[average_max_index]:.6f}")
print(f"Time of average maximum [s]   : {time[average_max_index]:.3f}")
print(f"Final average concentration   : {average_concentration[-1]:.6f}")

print("\n[TEMPERATURE]")
print(f"Initial temperature [K] : {temperature[0]:.3f}")
print(f"Final temperature [K]   : {temperature[-1]:.3f}")
print(f"Maximum temperature [K] : {temperature.max():.3f}")

print("\n[NUMERICAL CHECK]")
print(f"Contains NaN              : {np.isnan(concentration).any()}")
print(f"Contains Inf              : {np.isinf(concentration).any()}")
print(f"Global minimum            : {concentration.min():.6f}")
print(f"Global maximum            : {concentration.max():.6f}")

print("\n[POST-PROCESSING CHECKS]")
print(f"Mass balance satisfied     : {mass_balance}")
print(f"Maximum residual           : {np.max(np.abs(residual)):.6e}")
print(f"Temperature check passed : {temperature_passed}")
print(f"Contains NaN             : {temperature_checks['contains_nan']}")
print(f"Contains Inf             : {temperature_checks['contains_inf']}")
print(f"Positive temperature     : {temperature_checks['positive_temperature']}")
print(f"Within physical bounds   : {temperature_checks['within_bounds']}")
print(f"Physical bounds satisfied  : {physical_bounds}")

print()
print(f"Dataset samples            : {len(dataset['rho'])}")
print(f"Unique radial nodes        : {len(np.unique(dataset['rho']))}")
print(f"Unique time instants       : {len(np.unique(dataset['tau']))}")

print(f"Surface concentration size : {global_output['surface_concentration'].shape}")
print(f"Average concentration size : {global_output['average_concentration'].shape}")

print("\n[PINN TRAINING DATA]")
print(f"Input matrix shape (X)     : {PINN_X_training.shape}")
print(f"Target matrix shape (Y)    : {PINN_Y_training.shape}")

print("\nFirst training sample")
print(f"rho                        : {PINN_X_training[0,0]:.6f}")
print(f"tau                        : {PINN_X_training[0,1]:.6f}")

if PINN_X_training.shape[1] > 2:
    print(f"current                    : {PINN_X_training[0,2]:.6f}")

if PINN_X_training.shape[1] > 3:
    print(f"intercalation flux         : {PINN_X_training[0,3]:.6e}")

if PINN_X_training.shape[1] > 4:
    print(f"dimensionless flux         : {PINN_X_training[0,4]:.6e}")

print(f"target concentration       : {PINN_Y_training[0,0]:.6f}")

print("\n============================================================")
