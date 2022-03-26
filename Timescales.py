'''
This script will look at the timescales of different processes for
those very hard black hole binaries. In particular the interaction
timescale to eject a binary from a cluster, to ionise the binary and
the merger timescale
'''

import pandas as pd; import numpy as np; import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import ode
import os; import glob; import sys

pd.options.mode.chained_assignment = None  
plt.rcParams.update({'font.size': 12})
G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 

def find_dir():
        '''
        Finds the likely location for the petar data files to be stored
        and gives the option to autoselect them.

        Returns data directory as a string
        '''

        # Finding possible directories where data could be stored
        directories = glob.glob("COMPAS_Output*")

        # Create a dictionary to store the available directories and index vals
        directoryList = {str(i): directories[i] for i in range(len(directories))}

        # Print the available directories
        print("Possible Directories:\n")
        for key, val in directoryList.items():
                print(key, ":", val)

        # Asking what directory the data is stored in and giving a list of potential directories
        chooseDirectory = input("\nWhat directory is the data stored in?  ")
        if chooseDirectory in directoryList.keys():
                dataDirectory = directoryList[str(chooseDirectory)]

        elif os.path.exists(str(chooseDirectory)):
                dataDirectory = str(chooseDirectory)

        else:
                print("Could not find directory\n")
                print("Quitting")
                sys.exit()

        return dataDirectory

# Setting path to data and for plots
cwd = os.getcwd()
COMPAS_Results_path = find_dir()
outPlots = "Timescale plots"

# Making directory for plots to exist in.
outdir = os.path.join(cwd, COMPAS_Results_path, outPlots)
if not os.path.exists(outdir): os.mkdir(outdir)

Rsol = 6.9*(10**8.) #in meters
MyrsToSec = 3.15*(10**13.) #time in sec
tobs = 13*(10**3.)*MyrsToSec #Age of MilkyWay

Gsi =  6.6*10**-11. #garavitaional constant in SI
c = 3.*(10**8.) #velocity of light in seconds
AUtoRsol = 214.9 #AU to Rsol
Msol = 2.*(10**30) #Solar mass in kg
betaWithoutMass = (64./5.)*(Gsi**3.0)/(c**5.0)
daysToSeconds = 86400
GyrsToSec = MyrsToSec * 1000
YrsToSec = 3600*24*365.25
G_pcMsolyr = 4.49e-15 # Gravitational constant in pc^3/Msol/year

#----tdelay

#-- Choose ODE integrator
backend = 'dopri5'


def tdelay(ai,ei,m1,m2):
    
    l=len(ei)
    t_merger=np.array([])
    
    for i in range (l):
        a0 = ai[i]*Rsol
        m_1 = m1[i]*Msol
        m_2 = m2[i]*Msol
        e0=ei[i]
    
        c0Part1 = a0*(1. - e0**2.0)
        c0Part2 = (1.+(121./304.)*e0**2.)**(870./2299.)
        c0Part3 = e0**(12./19.)
        c0 = c0Part1/(c0Part2*c0Part3)
        beta = betaWithoutMass*m_1*m_2*(m_1+m_2)

        constant = (12./19.)*(c0**4.)/beta
        #print ((1. - e0**2.)**(3./2.))
    
        func = lambda e: constant*((e**(29./19.))*(1. + (121./304.)*e**2.)**(1181./2299.))/((1. - e**2.)**(3./2.))

        #-- Create ODE solver object
        solver = ode(func).set_integrator(backend)

        #-- Define initial and final parameters
        T0 = 0        #-- Initial value of T
        efinal = 1E-5 #-- Maximum value of e to integrate to

        solver.set_initial_value(T0, e0) #.set_f_params(r)

        sol = [] #-- Create an empty list to store the output in (here it will be the e list)
    
        #-- Define a function to append the output to our list
        def solout(e, T):
            sol.append([e, T/YrsToSec])
        solver.set_solout(solout)

        #-- This line actually integrates the ODE, no loop is required
        solver.integrate(efinal)
        
        #-- Convert list to array
        sol = np.array(sol, dtype=object)
  
        #-- Use sol to find the location
  
        e = sol[:, 0]
        T = np.abs(sol[:,1])

        t_max = max(np.abs(sol[:,1]))
        
        tm = t_max
        #print tm
        
        t_merger = np.append(t_merger, tm)

    return t_merger.T

def recoil_kick_timescale(mp, m1, m2, v_esc, x, n=1e4):
    '''
    This function calculates the timescale for a
    strong interaction that ejects the binary from
    a cluster of a given escape velocity. Note that
    x is the proportionality constant between escape velocity 
    and velocity dispersion. n is the number density of the cluster
    default is 10^4 pc^-3
    '''
    
    G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
    c = 3e5 #kms^-1
    
    # Convert n from pc^-3 into R_sol^-2 km^-1
    n *= 5.083e-16 * 3.241e-14 

    mtot = m1+m2 # M_sol

    tRK = (v_esc**3)/(x*n*np.pi*G**2*mp**2*(1+2*x*(mtot/mp))) # Seconds
    return tRK/(3600*24*365.25)

def interaction_timescale_v2(m1, m2 , semi, Mcl, rh):
    '''
    Calculates the interaction timescale in terms of the 
    relaxation time for the relaxation time of the cluster
    
    Input >>> m1, m2 = Binary primary and secondary mass (Msol)
              semi = Binary semi-major axis (Rsol)
              Mcl = Cluster mass (Msol)
              rh = Cluster half-mass radius (pc)
              
    Output >>> t3trh = interaction timescale per relaxation time
    '''
    rh *= 4.435e7 # Convert to Rsol

    t3trh = 10*(m1*m2)/Mcl**2 * rh/(2*semi) # Interaction timescale/relaxation time

    return t3trh

SN = pd.read_csv(os.path.join(cwd, COMPAS_Results_path, "BSE_Supernovae.csv"), skiprows=2)
SP = pd.read_csv(os.path.join(cwd, COMPAS_Results_path,"BSE_System_Parameters.csv"), skiprows=2)

invalidVals = SN.loc[(SN["SystemicSpeed "] == "          -nan")|(SN["SystemicSpeed "] == "          -nan")|(SN["SystemicSpeed "] == "          -nan")]
if len(invalidVals)>0:
    print("{} systems dropped".format(len(invalidVals)))
    SN.drop(invalidVals.index, inplace=True)

SN = SN.astype({"SystemicSpeed ":"float64", 
                "ComponentSpeed(SN)":"float64", 
                "ComponentSpeed(CP)":"float64",
                "SemiMajorAxis ":"float64",
                " Eccentricity ":"float64"})


# systems undergoing a second SN
SN_dup_1 = SN.loc[SN.duplicated(subset="    SEED    ", keep = "last")]

# BHBH systems that remain bound
BHB = SN.loc[(SN["Unbound"]==0)&(SN["Stellar_Type(SN)"]==14)&(SN["Stellar_Type(CP)"]==14)]
BHB.reset_index(drop=True, inplace=True)

# BHBH systems which become unbound
BHB_unbound = SN.loc[(SN["Unbound"]==1)&(SN["Stellar_Type(SN)"]==14)&(SN["Stellar_Type(CP)"]==14)]
BHB_unbound.reset_index(drop=True, inplace=True)

# Black holes formed in the first supernovae which break the binary
BH1_unbound = SN.loc[(SN["Unbound"]==1)&(SN["Stellar_Type(SN)"]==14)&(~SN["    SEED    "].isin(SN_dup_1["    SEED    "]))]
BH1_unbound.reset_index(drop=True, inplace=True)

##################################################
# Creating a variety of different clusters
Mcl_range = np.logspace(4, 6, 5, endpoint = True) # Range of cluster masses (Msol)
rh_range = np.linspace(1, 4, 5) # Range of half-mass radii (pc)

# Defining a 2D array for all the possible densities, escape velocities and relaxation times
trh_range = np.zeros((len(Mcl_range), len(rh_range)))
rhoh_range = np.zeros((len(Mcl_range), len(rh_range)))
v_esc_range = np.zeros((len(Mcl_range), len(rh_range)))

print('Relaxation times for a range of masses and radii\n')
for i in range(len(Mcl_range)):
    for j in range(len(rh_range)):
        '''
        Fills in the 2D array of densities, escape velocities and relaxation times, 
        for every combination of cluster mass and half-mass radius
        ''' 
        rhoh_range[i,j] = (Mcl_range[i]/2)/(4/3*np.pi*rh_range[j]**3) # rho range (Msol/pc^3)
        v_esc_range[i, j] = 50*(Mcl_range[i]/1e5)**(1/3)*(rhoh_range[i,j]/1e5)**(1/6) # Vesc range (km/s)

        trh_range[i, j] = 0.138/8.09*np.sqrt((Mcl_range[i]*rh_range[j])/G_pcMsolyr) # Relaxation time (years)

        # Printing result
        print('For Mcl = {0:.3g} Msol, rh = {1:.3g} pc:  Trh = {2:.3g} years'.format(Mcl_range[i], rh_range[j], trh_range[i,j]))
    print('----------------------------\n')

# Making subplots
fig, ax = plt.subplots(figsize=(8,6.5))

# Empty 2D array for the fractions merging before interaction
frac_merge_inside = np.zeros((len(Mcl_range), len(rh_range)))

for i in range(len(Mcl_range)):
    for j in range(len(rh_range)):
        # Ejected on first SN?
        retained_from_first = SN_dup_1.loc[SN_dup_1["SystemicSpeed "]<v_esc_range[i, j]]

        # Number of bound BHBH systems that are retained 
        retained_bound = BHB.loc[(BHB["SystemicSpeed "]<v_esc_range[i, j])&(BHB["    SEED    "].isin(retained_from_first["    SEED    "]))]

        # Now look at the number of retained lone BHs
        retained_unbound_first_mass = BH1_unbound["   Mass(SN)   "].loc[BH1_unbound["ComponentSpeed(SN)"]<v_esc_range[i, j]]
        retained_unbound_second_mass1 = BHB_unbound["   Mass(SN)   "].loc[(BHB_unbound["ComponentSpeed(SN)"]<v_esc_range[i, j])&(BHB_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]
        retained_unbound_second_mass2 = BHB_unbound["   Mass(CP)   "].loc[(BHB_unbound["ComponentSpeed(CP)"]<v_esc_range[i, j])&(BHB_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]

        sigma = v_esc_range[i, j]/4.77
        mu = (retained_bound["   Mass(SN)   "]*retained_bound["   Mass(CP)   "])/(retained_bound["   Mass(SN)   "]+retained_bound["   Mass(CP)   "]) # M_sol
        ah = G*mu/sigma**2 # R_sol
        ah_a = ah/retained_bound["SemiMajorAxis "]

        hard = retained_bound.loc[ah_a>1]

        T_GW = tdelay(hard["SemiMajorAxis "].values, hard[" Eccentricity "].values, hard["   Mass(CP)   "].values, hard["   Mass(SN)   "].values)
        T_GW /= trh_range[i, j]
        
        T_RK = interaction_timescale_v2(hard["   Mass(SN)   "].values, hard["   Mass(CP)   "].values, hard["SemiMajorAxis "].values, Mcl_range[i], rh_range[j]**(1/3))
        #T_RK *= trh[i]

        HubbleTime = 14e9/trh_range[i, j]

        # Caculating the fraction of systems with T_GW < T_RK
        frac_merge_inside[i, j] = sum((T_GW<T_RK)&(T_GW<HubbleTime))/sum(T_GW<HubbleTime)
        
    print('{0:1.0%} completed'.format((i+1)/len(Mcl_range)), end='\r', )

# Making a contour plot for the fraction of systems that merge before they have an interaction. 
ax.set_yscale('log')
im = ax.contourf(rh_range, Mcl_range, frac_merge_inside)

# Generating color bar and giving it a label
cbar = fig.colorbar(im, )
cbar.set_label('Fraction merge before interaction', rotation=270, labelpad=12)

# Formatting plot
ax.set_xlabel('Half-mass Radius (pc)')
ax.set_ylabel('Cluster mass (M$_{\odot}$)')
fig.tight_layout()

fig.savefig(os.path.join(outdir, 'Fraction merging before interaction.png'))
plt.show()
