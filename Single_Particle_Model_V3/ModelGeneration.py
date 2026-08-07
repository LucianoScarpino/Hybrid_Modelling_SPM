import numpy as np

class ModelGeometry(object):
    """Define the geometry and transport/thermal laws of the reference SPM.

    The class builds dimensionless grids and evaluates state-dependent solid
    diffusion and the lumped thermal balance used by the finite-volume solver.
    """
    def __init__(self,
                Rp,
                Ds,
                tf,
                tr,
                tload,
                radial_spacing,
                diffusion_type,
                T_ref,      
                T_amb,      
                m,          
                cp,         
                R_el,       
                h,          
                A_th,       
                Ea,         
                Rg,         
                 ):
        
        self.radius = Rp                                                    #physical
        self.diff_coeff = Ds                                                #physical, if type ='adaptive', it becames the Dref own by the problem
        self.adaptive_diff_coeff = Ds                                       #pyshical, initialization, if type = 'constant' remains Ds = Dref            
        self.final_time = tf                                                #pyshical
        self.ramp_time = tr                                                 #pyshical
        self.ramp_down = tload                                              #pyshical, starting time of ramp_down
        self.rad_spacing = radial_spacing                                   #pyshical
        self.diffusion_type = diffusion_type                    
        self.T_ref = T_ref                                                  #thermal                                           
        self.T_amb = T_amb                                                  #thermal
        self.mass = m                                                       #thermal    
        self.heat_cap = cp                                                  #thermal
        self.eq_resistence = R_el                                           #thermal
        self.convection_coeff = h                                           #thermal
        self.heat_area = A_th                                               #thermal
        self.activation_energy = Ea                                         #thermal
        self.Rg = Rg                                                        #thermal

    def diffusion_coefficient(self,concentration,C0,T,alpha=1.0):
        """Return nodal diffusivity for the current concentration and temperature."""
        if self.diffusion_type == 'adaptive':
            self.adaptive_diff_coeff = self.diff_coeff * np.exp(alpha * (concentration - C0)) * np.exp(-(self.activation_energy/self.Rg)*(1/T - 1/self.T_ref))
            return self.adaptive_diff_coeff

        return self.diff_coeff * np.ones_like(concentration)

    def get_diffusion_time(self):
        if np.any(self.adaptive_diff_coeff <= 0):
            raise ValueError("The diffusion coefficient must be positive.")

        return self.radius**2 / self.adaptive_diff_coeff

    def get_dimentionless_times(self):
        tauf = (self.diff_coeff * self.final_time)/(self.radius**2)             #final dimentionless time
        taur = (self.diff_coeff * self.ramp_time)/(self.radius**2)              #dimentionless ramp time
        taulaod = (self.diff_coeff * self.ramp_down)/(self.radius**2)           #dimentionless ramp down starting time

        return tauf,taur,taulaod

    def get_dimentionless_radius(self,N):
        """
        Uniform grid N+1 equidistant points.

        """

        return np.linspace(0,1,N+1)

    def compute_interfaces(self,N,rhos):
        """
        Compute interfaces used to compute volumes of control.

        """
        interfaces = np.zeros(N+2)
        interfaces[0] = 0
        interfaces[N+1] = 1

        for i in range(N):
            interfaces[i+1] = (rhos[i] + rhos[i+1])/2

        return interfaces

    def compute_control_volumes(self,N,interfaces):
        """
        Compute Volume of crontrols for each competent dimenstionless radius.
        
        """
        internal_volumes = np.zeros(N-1)    

        centre_volume = (interfaces[1]**3)/3
        surface_volume = (1 - interfaces[N]**3)/3

        for i in range(1,N):
            internal_volumes[i-1] = (interfaces[i+1]**3 - interfaces[i]**3)/3

        return centre_volume,surface_volume,internal_volumes

    def get_physical_sampling_interval(self,Ntau):
        return self.final_time/(Ntau-1)                     # --> dt

    def generate_time_grid(self,dt,Ntau):
        """
        Define interval points where require solver to compute solution.
         
        """
        grid = np.zeros(Ntau)

        for k in range(Ntau):
            grid[k] = k * dt

        return grid

    def get_tau(self,grid):
        tau = (grid * self.diff_coeff)/(self.radius**2)

        return tau

    def init_concentration(self,C0,N):
        return C0 * np.ones(N+1)

    def get_beta_q(self):
        return self.eq_resistence/(self.mass * self.heat_cap)

    def get_beta_h(self):
        return self.convection_coeff * self.heat_area/(self.mass * self.heat_cap)

    def compute_temperature_derivatives(self,current,temperature):
        """Return the dimensionless lumped-temperature derivative."""
        diffusion_time = self.radius**2 / self.diff_coeff
        betaQ = self.get_beta_q()
        betaH = self.get_beta_h()

        return diffusion_time * ((betaQ * current**2) - betaH * (temperature - self.T_amb)) 

class SurfaceFlux(object):
    """Map an applied-current profile to dimensional and normalized fluxes.

    It supports constant or ramp--hold--ramp operating profiles and supplies
    the surface boundary input used by the particle model.
    """
    def __init__(self,I1C,F,Sp,Rp,Ds,Cmax):
        self.nominal_current = I1C
        self.F = F                                                                          #Faraday constant
        self.active_surf = Sp                                                               #electroactive surface area
        self.radius = Rp
        self.diff_coeff = Ds
        self.max_concentration = Cmax

    def define_applied_current(self,t,tr,tload,profile='operating'):
        """Evaluate the selected current profile at physical time ``t``."""
        if profile == 'constant':
            curr = self.nominal_current
            
        elif profile == 'operating':
            if t < tr and t >= 0:
                curr = self.nominal_current/2 * (1 - np.cos(np.pi * t/tr))                      #current during rump-up phase
            elif t >= tr and t < tload:
                curr = self.nominal_current                                                     #current during constant-current phase
            elif t >= tload and t < (tload + tr):
                curr = self.nominal_current/2 * (1 + np.cos(np.pi * (t - tload)/tr))            #current during ramp-down phase
            else:
                curr = 0

        return curr

    def get_intercalation_flux(self,curr):
        return - curr/(self.F * self.active_surf)   # --> j(t(tau))

    def get_dimentionless_intercalation_flux(self,j):
        """Convert dimensional flux to its normalized value and nominal scale."""
        delta = (self.radius * j)/(self.diff_coeff * self.max_concentration)
        delta1C = self.nominal_current * self.radius/(
            self.diff_coeff * self.F * self.active_surf * self.max_concentration)           #nominal  dimentionless intercalation flux

        return delta,delta1C

    def get_current_profile(self,curr):
        return curr/self.nominal_current
