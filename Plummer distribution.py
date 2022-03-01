import numpy as np; from random import choices
import matplotlib.pyplot as plt; 

def Kroupa(N):
    '''
    Calculates N stellar masses drawing from a Kroupa IMF 0.08 < m < 130
    
    Input >>> N = number of stars wanted
    
    Output >>> masses = N-sized array of stellar masses
    '''

    # Create a list of potential masses and then calculate their weights by using Kroupa IMF
    potential_mass = np.logspace(np.log10(0.08), np.log10(130), 10**4, endpoint=True)

    weights_low = 0.204*potential_mass[np.where(potential_mass<0.5)]**(-1.3) # Probabilities below m=0.5Msol
    weights_high = 0.204*potential_mass[np.where(potential_mass>=0.5)]**(-2.3) # Probabilities above m=0.5M_sol

    weights_total = np.append(weights_low, weights_high)

    # Picking the final masses based on the weights 
    masses = choices(potential_mass, weights_total,k=N)
    
    return masses

masses = Kroupa(1000)

fig, ax = plt.subplots()

ax.hist(masses, bins=50, density =True, histtype='step')

plt.show()
