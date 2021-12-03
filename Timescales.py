'''
This script will look at the timescales of different processes for
those very hard black hole binaries. In particular the interaction
timescale to eject a binary from a cluster, to ionise the binary and
the merger timescale
'''

import pandas as pd; import numpy as np; import scipy as sp
from random import choices; import matplotlib.pyplot as plt
import os 

pd.options.mode.chained_assignment = None  

G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
cwd = os.getcwd()

def GW_timescale(a, e, m1, m2):
    '''
    This function calculates the GW timescale using
    equation (16) from Peters & Mathews 1963.
    Note that this includes up to 2.5 PN terms
    '''
    
    G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
    c = 3e5 #kms^-1
    
    mtot = m1+m2 # M_sol
    mu = (m1*m2)/mtot # M_sol

    tGW = (5/64)*((c**5)*((a**4)*695700))/((G**3)*mu*(mtot**2))*((1-e**2)**(7/2))/(1+(73*e**2)/24 + (37*e**4)/96) # Seconds
    return tGW/(3600*24*365.25)

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

COMPAS_Results_path = "/COMPAS_Output_1%solar_metallicity"
SN = pd.read_csv(cwd + COMPAS_Results_path + "/BSE_Supernovae.csv", skiprows=2)
SP = pd.read_csv(cwd + COMPAS_Results_path + "/BSE_System_Parameters.csv", skiprows=2)

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

v_esc = np.array([25, 50, 75, 100, 125]) # Cluster escape velocity kms^-1

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

    T_GW = GW_timescale(hard["SemiMajorAxis "], hard[" Eccentricity "], hard["   Mass(CP)   "], hard["   Mass(SN)   "])
    T_RK = recoil_kick_timescale(hard["PerturbingMass"], hard["   Mass(CP)   "], hard["   Mass(SN)   "], v_esc[i], 4.77)

    # T_GW = TRK line

    #T_RK_array = np.logspace(np.log10(0.95*min(T_RK)), np.log10(1.05*max(T_RK)), 500)
    T_RK_array = np.linspace(0.95*min(T_RK), 1.05*max(T_RK), 500)
    T_GW_array = T_RK_array

    # Plotting the results
    plt.figure(figsize=(6,5))
    plt.loglog(T_GW, T_RK, 'k.', alpha = 0.8, zorder = 1)

    plt.vlines(14e9, 0.95*min(T_RK), 1.05*max(T_RK), colors='red', linestyles='-.', label = "Hubble time", zorder=3)
    plt.loglog(T_GW_array, T_RK_array, '--', zorder = 2, label = "$\\tau_{GW} = \\tau_{RK}$")

    plt.title("GW timescale and recoil timescale for $v_{{esc}}={0}$ and a perturber mass pulled from mass distribution".format(v_esc[i]))
    plt.ylim(0.95*min(T_RK), 1.05*max(T_RK))
    plt.xlabel("Merger timescale (years)")
    plt.ylabel("Recoil kick timescale (years)")
    plt.legend(loc="best")

plt.show()