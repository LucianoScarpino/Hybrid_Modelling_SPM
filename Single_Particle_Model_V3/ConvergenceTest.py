import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from ParticleSimulation import Simulate
from matplotlib.ticker import ScalarFormatter
from DatasetGenerator import PostProcessing
from Config import parse_params

args = parse_params()
args.Ds_type = 'constant'   #convergence check only on constant mode
n = [25,50,100,200,400]     #check convergence independently from mesh

average_concentrations = {}
surface_concentration = {}
times = {}
profiles = {}
radii = {}


for i in n: 
    args.N = i
    simulation = Simulate(args)
    solutions = simulation.run()

    tau = solutions["tau"]
    time = solutions["time"]
    radius = solutions["radius"]
    concentration = solutions["concentration"]

    surface_concentration[i] = concentration[-1,:]
    times[i] = time 
    profiles[i] = concentration[:,-1]
    radii[i] = radius

    post = PostProcessing(**solutions)
    average_concentration = post.compute_average_concentration()
    average_concentrations[i] = average_concentration

data = []
Cs_ref = surface_concentration[400]
Cavg_ref = average_concentrations[400]

for key in n:
    cs_error = np.linalg.norm(
        surface_concentration[key] - Cs_ref,
        ord=2
    )

    cavg_error = np.linalg.norm(
        average_concentrations[key] - Cavg_ref,
        ord=2
    )

    data.append({
        "N": key,
        "||Cs - Cs_ref||_2": cs_error,
        "||Cavg - Cavg_ref||_2": cavg_error
    })

table = pd.DataFrame(data)
table.to_csv('./SPMDataset/mesh_convergence_results.csv')

print(table.to_string(index=False))

fig,axs = plt.subplots(3,1,figsize=(8,8))
for i in n:
    axs[0].plot(
        times[i],
        surface_concentration[i],
        label=f"N = {i}"
    )

axs[0].set_title("Surface concentration convergence")
axs[0].set_xlabel("Time [s]")
axs[0].set_ylabel(r"$C(\rho=1,t)$")
axs[0].grid(True)
axs[0].legend()

for i in n:
    axs[1].plot(
        times[i],
        average_concentrations[i],
        label=f"N = {i}"
    )

axs[1].set_title("Average concentration convergence")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel(r"$\bar{C}(t)$")
axs[1].grid(True)
axs[1].legend()

for i in n:
    axs[2].plot(
        radii[i],
        profiles[i],
        linewidth=1.5,
        label=f'N = {i}'
    )

axs[2].set_title("Final radial concentration profile")
axs[2].set_xlabel(r"Dimensionless radius $\rho$")
axs[2].set_ylabel(r"$C(\rho,t_f)$")
axs[2].grid(True)
axs[2].legend()

formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
axs[2].yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig('./Images/mesh_comparison.png')
plt.show()