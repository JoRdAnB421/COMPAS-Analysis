'''
This script is for checking that I have correctly derived an expression 
for the kick required to break a binary.
'''

import pandas as pd
import numpy as np

G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 

def min_kick(M1, M2, a):
    '''
    Derives the minimum kick required to
    break a binary from the expression:
    v_kick = sqrt(G(M1+M2)/a)
    '''
    return np.sqrt(G*(M1+M2)/a)

# Setting the path to the COMPAS results 
COMPAS_Results_path = r"C:\Users\jorda\OneDrive\Desktop\PhD\COMPAS Results\COMPAS_Output_0.0001_solar_metallicity"
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

minimum_kicks = min_kick(SN["   Mass(SN)   "], SN["   Mass(CP)   "], SN["SemiMajorAxis<SN"])

print("The number of times my expression is correct: {}".format(SN["Applied_Kick_Magnitude(SN)"].loc[(SN["Applied_Kick_Magnitude(SN)"]>minimum_kicks)&(SN["Unbound"] == 1)].count() + SN["Applied_Kick_Magnitude(SN)"].loc[(SN["Applied_Kick_Magnitude(SN)"]<minimum_kicks)&(SN["Unbound"] == 0)].count()))
print("The number of times my expression is incorrect: {}".format(SN["Applied_Kick_Magnitude(SN)"].loc[(SN["Applied_Kick_Magnitude(SN)"]<minimum_kicks)&(SN["Unbound"] == 1)].count() + SN["Applied_Kick_Magnitude(SN)"].loc[(SN["Applied_Kick_Magnitude(SN)"]>minimum_kicks)&(SN["Unbound"] == 0)].count()))
#print(SN["Stellar_Type(SN)"].loc[(SN["Applied_Kick_Magnitude(SN)"]<minimum_kicks)&(SN["Unbound"] == 1)])