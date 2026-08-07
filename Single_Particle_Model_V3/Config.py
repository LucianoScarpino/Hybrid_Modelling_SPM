from argparse import ArgumentParser

def parse_params():
    """
    Parse command-line arguments.
    """

    parser = ArgumentParser(
        description="Single Particle Model simulation."
    )

    # Physical parameters
    parser.add_argument("--Rp", type=float, default=8.5e-6,
                        help="Particle radius [m]")

    parser.add_argument("--Sp", type=float, default=1.167,
                        help="Particle surface area [m^2]")

    parser.add_argument("--Ds_type", type=str, default='adaptive',
                        help="Solid diffusion behavior", choices=['adaptive','constant'])

    parser.add_argument("--Ds", type=float, default=1.213e-14,
                        help="Solid diffusion coefficient [m^2/s]")

    parser.add_argument("--alpha", type=float, default=1.0)                        
    
    parser.add_argument("--F", type=float, default=96487,
                        help="Faraday constant [C/mol]")

    parser.add_argument("--cs_max", type=float, default=51410,
                        help="Maximum lithium concentration [mol/m^3]")

    # Initial condition
    parser.add_argument("--C0", type=float, default=0.5,
                        help="Initial normalized concentration")

    # Current profile
    parser.add_argument("--I1C", type=float, default=1.656,
                        help="Nominal current [A]")

    parser.add_argument("--profile", type=str, default='operating',
                        choices=["operating","constant"], help="type of current profile")

    parser.add_argument("--tr", type=float, default=300,
                        help="Current ramp duration [s]")

    parser.add_argument("--tload", type=float, default=3600,
                        help="Constant-current duration [s]")

    parser.add_argument("--tf", type=float, default=6000,
                        help="Final simulation time [s]")

    # Thermal parameters
    parser.add_argument("--T_ref", type=float, default=298.15,
                        help="Reference temperature [K]")

    parser.add_argument("--T_amb", type=float, default=298.15,
                        help="Ambient temperature [K]")

    parser.add_argument("--Ea", type=float, default=35000.0,
                        help="Activation energy for diffusion [J/mol]")

    parser.add_argument("--Rg", type=float, default=8.314462618,
                        help="Universal gas constant [J/(mol K)]")

    parser.add_argument("--m", type=float, default=0.045,
                        help="Equivalent cell mass [kg]")

    parser.add_argument("--cp", type=float, default=900.0,
                        help="Equivalent specific heat capacity [J/(kg K)]")

    parser.add_argument("--R_el", type=float, default=0.10,
                        help="Equivalent electrical resistance [Ohm]")

    parser.add_argument("--h", type=float, default=10.0,
                        help="Convective heat transfer coefficient [W/(m^2 K)]")

    parser.add_argument("--A_th", type=float, default=0.01,
                        help="Equivalent heat exchange area [m^2]")

    # Numerical parameters
    parser.add_argument("--N", type=int, default=100,
                        help="Number of radial intervals")

    parser.add_argument("--Ntau", type=int, default=201,
                        help="Number of stored time instants")

    parser.add_argument("--abs_tol", type=float, default=1e-8,
                        help="Absolute tolerance")

    parser.add_argument("--rel_tol", type=float, default=1e-6,
                        help="Relative tolerance")

    parser.add_argument("--ODE_solver",choices=["solve_ivp"],
                        default="solve_ivp", help="ODE integration method")

    # Output parameters
    parser.add_argument("--variant_number", type=int, default=None,
                        help="Variant number used to save a separate dataset")

    return parser.parse_args()
