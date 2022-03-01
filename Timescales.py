'''
This script will look at the timescales of different processes for
those very hard black hole binaries. In particular the interaction
timescale to eject a binary from a cluster, to ionise the binary and
the merger timescale
'''

from pickle import TUPLE3
import pandas as pd; import numpy as np; import scipy as sp
from random import choices; import matplotlib.pyplot as plt
from scipy.integrate import ode
import os 

pd.options.mode.chained_assignment = None  
plt.rcParams.update({'font.size': 12})
G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 

# Setting path to data and for plots
cwd = os.getcwd()
COMPAS_Results_path = "COMPAS_Output_1%solar_metallicity"
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


#----tdelay

#-- Choose ODE integrator
backend = 'dopri5'


def tdelay(ai,ei,m1,m2):
    
    l=len(ei)
    t_merger=[]
    
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
        
        t_merger.append(tm)

    return t_merger

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

def interaction_timescale(m1, m2, semi, v_esc, rhoh = 1200):
    '''
    Calculates the timescale for and interaction from eq (13) and eq (20) of
    Antonini & Gieles (2020)
    
    Input >>> m1, m2 = binary primary and secondary mass (Msol)
              semi = binary semi-major axis (Rsol)
              v_esc = cluster escape velocity (km/s)
              rhoh = cluster density within half-mass radius (Msol/pc^3)
    
    Output >>> t3 = timescale for interaction (years)
              '''
    G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2

    rhoh *= 1.146e-23 # Convert rhoh to Msol/Rsol^3

    # Find binary component
    binary_comp = (G*m1*m2)/(semi*0.809)

    # Find cluster component
    cluster_comp = (3*G)/(8*np.pi*rhoh*v_esc**4)

    # Calculate timescale and return
    t3 = 0.069*binary_comp*cluster_comp**(0.5) # Units of secs Rsol/km

    # Convert to years
    t3 *= 6.957e5/(3600*24*365)

    return abs(t3)

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

v_esc = np.array([5, 10, 25, 50, 80]) # Cluster escape velocity kms^-1

rhoh=1200 # M_sol/pc

# Making subplots
fig, ax = plt.subplots(figsize=(8,6.5))

# Empty list to find max and min for straight lines later
RKrange = [10**20, 0]
for i in range(len(v_esc)):
    # Ejected on first SN?
    retained_from_first = SN_dup_1.loc[SN_dup_1["SystemicSpeed "]<v_esc[i]]

    # Number of bound BHBH systems that are retained 
    retained_bound = BHB.loc[(BHB["SystemicSpeed "]<v_esc[i])&(BHB["    SEED    "].isin(retained_from_first["    SEED    "]))]

    # Now look at the number of retained lone BHs
    retained_unbound_first_mass = BH1_unbound["   Mass(SN)   "].loc[BH1_unbound["ComponentSpeed(SN)"]<v_esc[i]]
    retained_unbound_second_mass1 = BHB_unbound["   Mass(SN)   "].loc[(BHB_unbound["ComponentSpeed(SN)"]<v_esc[i])&(BHB_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]
    retained_unbound_second_mass2 = BHB_unbound["   Mass(CP)   "].loc[(BHB_unbound["ComponentSpeed(CP)"]<v_esc[i])&(BHB_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]

    lone_mass = np.append(retained_unbound_first_mass.values, retained_unbound_second_mass1.values)
    lone_mass = np.append(lone_mass, retained_unbound_second_mass2)

    values, bins = np.histogram(lone_mass, bins = range(0, round(max(lone_mass)), 1), density = True)
    bin_mid = np.array([(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)])
    
    retained_bound["PerturbingMass"] = choices(bin_mid, weights=values, k=len(retained_bound))

    sigma = v_esc[i]/4.77
    mu = (retained_bound["   Mass(SN)   "]*retained_bound["   Mass(CP)   "])/(retained_bound["   Mass(SN)   "]+retained_bound["   Mass(CP)   "]) # M_sol
    ah = G*mu/sigma**2 # R_sol
    ah_a = ah/retained_bound["SemiMajorAxis "]

    hard = retained_bound.loc[ah_a>1]

    T_GW = tdelay(hard["SemiMajorAxis "].values, hard[" Eccentricity "].values, hard["   Mass(CP)   "].values, hard["   Mass(SN)   "].values)
    #T_RK = recoil_kick_timescale(hard["PerturbingMass"], hard["   Mass(CP)   "], hard["   Mass(SN)   "], v_esc[i], 4.77)

    T_RK = interaction_timescale(hard["   Mass(SN)   "].values, hard["   Mass(CP)   "].values, hard["SemiMajorAxis "], v_esc[i], rhoh=rhoh)

    # T_GW = TRK line
    if min(T_RK) < RKrange[0]:
        RKrange[0]=min(T_RK)
    if max(T_RK) > RKrange[1]:
        RKrange[1]=max(T_RK)

    # Plotting the results
    #plt.figure(figsize=(6,5))
    ax.loglog(T_GW, T_RK, '.', alpha = 0.8, zorder = 1, label = "$v_{{esc}}$ = {} kms$^{{-1}}$".format(v_esc[i]))

    #plt.title("GW timescale and recoil timescale for $v_{{esc}}={0}$ and a perturber mass\npulled from mass distribution".format(v_esc[i]))
    #ax.set_ylim(0.95*min(T_RK), 1.05*max(T_RK))
    #plt.xlabel("Merger timescale (years)")
    #plt.ylabel("Recoil kick timescale (years)")
    

    #savename = "GWvsRK for v_esc = {}.png".format(v_esc[i])


#T_RK_array = np.logspace(np.log10(0.95*min(T_RK)), np.log10(1.05*max(T_RK)), 500)

# Defining and plotting a hubble time and T_RK=T_GW
T_RK_array = np.linspace(0.95*RKrange[0], 1.05*RKrange[1], 500)
T_GW_array = T_RK_array


ax.vlines(14e9, 0.8*RKrange[0], 1.2*RKrange[1], colors='black', linestyles=':', label = "Hubble time", zorder=3)
ax.loglog(T_GW_array, T_RK_array, '--', color='black', zorder = 2, label = "$\\tau_{GW} = \\tau_{RK}$")

# Making a single legend for all 
'''handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
'''
# Adding text for labels
ax.set_xlabel('Merger timescale (years)')
ax.set_ylabel('Recoil kick timescale (years)')
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles[::-1], labels[::-1], loc='upper right')
#fig.text(0.05, 0.5, 'Recoil kick timescale (years)', va='center', ha='center', rotation=90)


fig.savefig(os.path.join(outdir, 'Timescale_plots.pdf'), dpi=100)

plt.show()
