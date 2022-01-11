import pandas as pd; import numpy as np; import matplotlib.pyplot as plt

'''
This script will plot the distribution of the applied kick magnitudes for the first and second supernovae in a system'''


COMPAS_Results_path = r"C:\Users\jorda\OneDrive\Desktop\PhD\COMPAS Results\COMPAS_Output_1%solar_metallicity"
SN = pd.read_csv((COMPAS_Results_path + r"\BSE_Supernovae.csv"), skiprows=2)
SP = pd.read_csv((COMPAS_Results_path + r"\BSE_System_Parameters.csv"), skiprows=2)

EAB = SP.loc[SP['Equilibrated_At_Birth'] == 1]
SN.drop(SN.loc[SN["    SEED    "].isin(EAB["    SEED    "])].index, inplace = True)

invalidVals = SN.loc[(SN["SystemicSpeed "] == "          -nan")|(SN["SystemicSpeed "] == "          -nan")|(SN["SystemicSpeed "] == "          -nan")]
if len(invalidVals)>0:
    print("{} systems dropped".format(len(invalidVals)))
    SN.drop(invalidVals.index, inplace=True)
SN = SN.astype({"SystemicSpeed ":"float64", 
                "ComponentSpeed(SN)":"float64", 
                "ComponentSpeed(CP)":"float64",
                "SemiMajorAxis ":"float64"})


BHB = SN.loc[(SN["Unbound"]==0)&(SN["Stellar_Type(SN)"]==14)&(SN["Stellar_Type(CP)"]==14)]
BHB.reset_index(drop=True, inplace=True)

SN_dup_1 = SN.loc[SN.duplicated(subset="    SEED    ", keep = "last")]

first_kick = SN_dup_1["Applied_Kick_Magnitude(SN)"].loc[SN["    SEED    "].isin(BHB["    SEED    "])]

print("did this work")

"""
BHB_unbound = SN.loc[(SN["Unbound"]==1)&(SN["Stellar_Type(SN)"]==14)&(SN["Stellar_Type(CP)"]==14)]
BHB_unbound.reset_index(drop=True, inplace=True)

plt.hist(BHB["Applied_Kick_Magnitude(SN)"], cumulative=True, histtype="step", linestyle = "-", label = "Second Kick")
plt.hist(first_kick, cumulative = True, histtype="step", label = "First Kick")
plt.legend(loc="best")
"""
"""

plt.figure()
plt.hist(BHB["Applied_Kick_Magnitude(SN)"], bins = np.logspace(np.log10(0.01), np.log10(600), 25), density=False, histtype="step", linestyle = "-", label = "Second Kick")
plt.hist(first_kick, bins = np.logspace(np.log10(0.01), np.log10(600), 25), density=False, histtype="step", label = "First Kick")

plt.title("First and second applied kick distributions")
plt.legend(loc="best")
plt.xscale("log")
plt.ylabel("PDF")
plt.xlabel("$vel \ kms^{-1}$")
"""
plt.show()