import numpy as np

from ModelGeneration import ModelGeometry,SurfaceFlux
from Assembler import FiniteVolumeSolver


class Simulate(object):
    """Orchestrate one coupled finite-volume particle simulation.

    Parsed settings are converted into geometry, current, diffusion, and
    thermal components; ``run`` returns the complete space--time solution.
    """

    def __init__(self,args):
        self.args = args
    
    def get_parameters(self):
        """Return parsed command-line settings as a simulation dictionary."""
        # Physical settings
        #Rp = 8.5 * 1e-6                 #particle radius                    [m]
        #Sp = 1.167                      #particle surface area              [m^2]
        #if Ds_type == 'constant'
            #Ds = 1.213 * 1e-14          #solid diffusion coefficient        [m^2/s]
        #if Ds_type = 'adaptive'
            #Ds = D0 * exp(alpha*(C-C0))
            #alpha = 1.0 (default)
            #D0 = 1.213 * 1e-14
        #F = 96487                       #Faraday constant                   [C/mol]
        #cs_max = 51410                  #maximum lithium concentration      [mol/m^3]

        #T_ref,      
        #T_amb,      # ambient temperature [K]
        #m,          # equivalent cell mass [kg]
        #cp,         # equivalent specific heat capacity [J/(kg K)]
        #R_el,       # equivalent electrical resistance [Ohm]
        #h,          # convective heat transfer coefficient [W/(m² K)]
        #A_th,       # equivalent heat exchange area [m²]
        #Ea,         # activation energy for lithium diffusion [J/mol]
        #Rg,         # universal gas constant [J/(mol K)]

        # Simulation setting
        #C0 = 0.5                        #initial normalized concentration   [-]
        #I1C = 1.656                     #nominal current (1C)               [A]
        #tr = 300                        #current ramp duration              [s]
        #tload = 3600                    #constant-current duration          [s]
        #tf = 6000                       #final simulation time              [s]

        #N = 100                         #num radial intervals
        #radial_spacing = 1/N            #radial spacing
        #Ntau = 201                      #num stored time instants
        #abs_tol = 1e-8                  #absolute tolerance
        #rel_tol = 1e-6                  #relative tolerance

        #ODE_solver = 'solve_ivp'

        radial_spacing = 1 / self.args.N

        return {
            "Rp": self.args.Rp,
            "Sp": self.args.Sp,
            "Ds_type": self.args.Ds_type,
            "Ds": self.args.Ds,
            "alpha": self.args.alpha,
            "F": self.args.F,
            "cs_max": self.args.cs_max,
            "C0": self.args.C0,
            "T_ref": self.args.T_ref,
            "T_amb":self.args.T_amb,
            "Ea": self.args.Ea,
            "Rg": self.args.Rg,
            "m":self.args.m,
            "cp":self.args.cp,
            "R_el":self.args.R_el,
            "h":self.args.h,
            "A_th":self.args.A_th,
            "profile":self.args.profile,
            "I1C": self.args.I1C,
            "tr": self.args.tr,
            "tload": self.args.tload,
            "tf": self.args.tf,
            "N": self.args.N,
            "radial_spacing": radial_spacing,
            "Ntau": self.args.Ntau,
            "abs_tol": self.args.abs_tol,
            "rel_tol": self.args.rel_tol,
            "ODE_Solver": self.args.ODE_solver
            }
    
    def run(self):
        """Run the coupled concentration--temperature model.

        Returns
        -------
        dict
            Grids, fields, flux histories, temperatures, and initial data.
        """
        params = self.get_parameters()

        geometry = ModelGeometry(
            Rp=params['Rp'],
            Ds=params['Ds'],
            diffusion_type= params['Ds_type'],
            tf=params['tf'],
            tr=params['tr'],
            tload=params['tload'],
            radial_spacing=params['radial_spacing'],
            T_ref=params['T_ref'],
            T_amb=params['T_amb'],
            m=params['m'],
            cp=params['cp'],
            R_el=params['R_el'],
            h=params['h'],
            A_th=params['A_th'],
            Ea=params['Ea'],
            Rg=params['Rg']
            )
        flux = SurfaceFlux(
            I1C=params['I1C'],
            F=params['F'],
            Sp=params['Sp'],
            Rp=params['Rp'],
            Ds=params['Ds'],
            Cmax=params['cs_max']
        )

        solver = FiniteVolumeSolver()
        
        N = params['N']
        Ntau = params['Ntau']

        tD = geometry.get_diffusion_time()

        rhos = geometry.get_dimentionless_radius(N=N)
        interfaces = geometry.compute_interfaces(N=N,rhos=rhos)
        _,surface_volume,internal_volumes = geometry.compute_control_volumes(N=N,interfaces=interfaces)

        dt = geometry.get_physical_sampling_interval(Ntau=Ntau)
        t = geometry.generate_time_grid(dt=dt,Ntau=Ntau)
        tau = geometry.get_tau(grid=t)

        currents = np.zeros_like(tau)
        intercalation_fluxes = np.zeros_like(tau)
        dimensionless_fluxes = np.zeros_like(tau)
        concentrations = geometry.init_concentration(N=N,C0=params['C0'])

        initial_state = np.concatenate((concentrations,
                                        np.array([params['T_ref']])))

        def rhs(tau_current,state):
            physical_time = tau_current* tD
            concentration_state = state[:-1]
            T_state = state[-1]

            D_nodes = geometry.diffusion_coefficient(
                concentration=concentration_state,
                T = T_state,
                C0=params["C0"],
                alpha=params['alpha']
                )
            
            normalized_D_nodes = D_nodes/params['Ds']
            current = flux.define_applied_current(t=physical_time,tr=params["tr"],tload=params["tload"],profile=params["profile"])
            intercalation_flux = flux.get_intercalation_flux(current)
            dimensionless_flux, _ = flux.get_dimentionless_intercalation_flux(intercalation_flux)

            dC_dtau = solver.assemble_diffusion_derivatives(
                N=params['N'],
                concentrations=concentration_state,
                interfaces=interfaces,
                boundary_flux=dimensionless_flux,
                internal_volumes=internal_volumes,
                surface_volume=surface_volume,
                diffusion_coeff = normalized_D_nodes
                )

            dT_dtau = geometry.compute_temperature_derivatives(
                current = current,
                temperature = T_state
            )

            if params['Ds_type'] == 'constant':
                dT_dtau = 0.0

            return solver.assemble_derivatives(
                concentration_deivatives=dC_dtau,
                temperature_derivatives=dT_dtau
            )

            
        tau_solution, state_solution = solver.solve_integration(
                                            right_hand=rhs,
                                            initial_state=initial_state,
                                            dimensionless_interval=(tau[0], tau[-1]),
                                            sampling_grid=tau,
                                            atol=params["abs_tol"],
                                            rtol=params["rel_tol"]
                                            )

        concentration_field = state_solution[:-1,:]
        temperatures = state_solution[-1,:]

        for k, tau_k in enumerate(tau_solution):
            t_ = tau_k * tD
            current = flux.define_applied_current(t_,params['tr'],params['tload'],profile=params['profile'])
            j = flux.get_intercalation_flux(current)
            delta, _ = flux.get_dimentionless_intercalation_flux(j)
            currents[k] = current
            intercalation_fluxes[k] = j
            dimensionless_fluxes[k] = delta

        return {
                "tau": tau_solution,
                "time": t,
                "radius": rhos,
                "concentration": concentration_field,
                'currents': currents,
                'intercalation_fluxes': intercalation_fluxes,
                'dimensionless_fluxes': dimensionless_fluxes,
                'initial_concentration': params['C0'],
                "temperatures":temperatures
                }
