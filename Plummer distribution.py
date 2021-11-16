import numpy as np
import matplotlib.pyplot as plt; 

def density_profile(r, a, M0):
    '''
    Creates a normalised density profile for the 
    Plummer model.
    
    Inputs >>> r = radius [R_sol]
               a = Plummer radius [R_sol]
               M0 = total cluster mass [M_sol]
    
    Output >>> rho = Density probability probability [M_sol/R_sol^3]
    '''

    G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
    A = 2*np.pi*a**2/M0 # Normalisation constant
    return A*(3*M0)/(4*np.pi*a**3)*(1+r**2/a**2)**(-5/2)


M0 = 10**5 # Sols

r = np.logspace(0, 4, 1000) 
a = 10**2

rho = density_profile(r, a, M0)
theta = np.linspace(0, np.pi, 1000)
phi = np.linspace(0, 2*np.pi, 1000)


rho = density_profile(r, a, M0)

plt.semilogx(r, rho)
plt.vlines(a, 0,  max(rho), linestyles='--')
plt.show()

