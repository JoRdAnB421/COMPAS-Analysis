'''
This script is for testing the curve fitting of the perturber mass 
'''

import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
import os; from random import choices

def gauss(x, h, mu, sigma):
    return h*np.exp(-((x-mu)**2)/(2*sigma**2))

G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
cwd = os.getcwd()

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

v_esc = np.array([20]) # Cluster escape velocity kms^-1
i=0

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

values, bins,_ = plt.hist(lone_mass, bins = range(0, round(max(lone_mass)), 1), histtype = "step", density = True)
bin_mid = np.array([(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)])

m_perturb = choices(bin_mid, weights=values, k=len(retained_bound))

print(len(m_perturb), len(retained_bound))


plt.show()
