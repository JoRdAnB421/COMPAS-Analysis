import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setting the path to the COMPAS results 
COMPAS_Results_path = r"C:\Users\jorda\OneDrive\Desktop\PhD\COMPAS Results\COMPAS_Output_OPTIMISTIC_RLOF"
SN = pd.read_csv((COMPAS_Results_path + r"\BSE_Supernovae.csv"), skiprows=2)
SP = pd.read_csv((COMPAS_Results_path + r"\BSE_System_Parameters.csv"), skiprows=2)



EAB = SP.loc[SP['Equilibrated_At_Birth'] == 1]
SP.drop(SP[SP["    SEED    "].isin(EAB["    SEED    "])].index, inplace = True)

invalidVals = SN.loc[(SN["SystemicSpeed "] == "          -nan")|(SN["SystemicSpeed "] == "          -nan")|(SN["SystemicSpeed "] == "          -nan")]
if len(invalidVals)>0:
    print("{} systems dropped".format(len(invalidVals)))
    SN.drop(invalidVals.index, inplace=True)
    SN = SN.astype({"SystemicSpeed ":"float64", 
                    "ComponentSpeed(SN)":"float64", 
                    "ComponentSpeed(CP)":"float64",
                    "SemiMajorAxis ":"float64"})

'''
def q(m1, m2):
    if m1>m2:
        q = m2/m1
    else:
        q = m1/m2
    return q



MR = np.array([])
for i in range(len(SP)):
    MR = np.append(MR, q(SP[" Mass@ZAMS(1) "].loc[i], SP[" Mass@ZAMS(2) "].loc[i]))
'''

MR = SP[" Mass@ZAMS(2) "]/SP[" Mass@ZAMS(1) "]

plt.hist(MR, bins = 25, histtype="step")
plt.title("Initial mass ratio distribution")
plt.xlabel("$q=\\frac{M_{2}}{M_{1}}$")

plt.show()