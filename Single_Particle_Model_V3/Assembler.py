import numpy as np
from scipy.integrate import solve_ivp

class FiniteVolumeSolver(object):
    def __init__(self):
        pass

    def assemble_diffusion_derivatives(self,
                            N,
                            concentrations,
                            interfaces,
                            boundary_flux,
                            internal_volumes,
                            surface_volume,
                            diffusion_coeff
                            ):
        drho = 1/N
        D_interface = 0.5 * (diffusion_coeff[:-1] + diffusion_coeff[1:])
        
        devs_C = np.zeros(N + 1)                                                            
        devs_C[0] = 6 * D_interface[0] * (concentrations[1] - concentrations[0])/(drho**2)

        for i in range(1,N):
            right_interface = ((D_interface[i] * interfaces[i+1]**2) * (concentrations[i+1] - concentrations[i]))/drho
            left_interface = ((D_interface[i-1] * interfaces[i]**2) * (concentrations[i] - concentrations[i-1]))/drho
            devs_C[i] = (right_interface - left_interface)/internal_volumes[i-1]

        surf_volume_interface = (D_interface[-1] * interfaces[-2]**2) * (concentrations[-2] - concentrations[-1])/drho
        devs_C[-1] = (surf_volume_interface - boundary_flux)/surface_volume
        right_hand = devs_C

        return right_hand               #f(tau,C)

    def assemble_derivatives(
            self,
            concentration_deivatives,
            temperature_derivatives
            ):

        return np.concatenate((concentration_deivatives,[temperature_derivatives]))

    def solve_integration(self,
                        right_hand,
                        initial_state,
                        dimensionless_interval,
                        sampling_grid,
                        atol,
                        rtol):

        solution = solve_ivp(
            fun=right_hand,
            t_span=dimensionless_interval,
            y0=initial_state,
            t_eval=sampling_grid,
            atol=atol,
            rtol=rtol,
            method="BDF"
        )

        if not solution.success:
            raise RuntimeError(solution.message)

        return solution.t, solution.y