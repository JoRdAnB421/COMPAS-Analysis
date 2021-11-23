'''
This script will look at the timescales of different processes for
those very hard black hole binaries. In particular the interaction
timescale to eject a binary from a cluster, to ionise the binary and
the merger timescale
'''

import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
import os 

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

    tGW = 5/64*((c**5)*a**4)/(G**3*mu*mtot**2)*((1-e**2)**(7/2))/(1+(73*e**2)/24 + (37*e**4)/96) # Seconds
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
    
    # Convert n into solar radii
    n = n*(4.435e7)**-3

    mtot = m1+m2 # M_sol

    tRK = (v_esc**3)/(x*n*np.pi*G**2*mp**2)*(1+2*x*mtot/mp)**-1
    return 

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
                "SemiMajorAxis ":"float64"})

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

v_esc = np.array([50]) # Cluster escape velocity kms^-1

T_GW = GW_timescale(BHB["SemiMajorAxis "], BHB[" Eccentricity "], BHB["   Mass(CP)   "], BHB["   Mass(SN)   "])
T_RK = recoil_kick_timescale(5, BHB["   Mass(CP)   "], BHB["   Mass(SN)   "], v_esc[0], 4.77)