import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
import os

dataDir = 'COMPAS_Output_solar_metallicity'

# Read in the data
SP = pd.read_csv(os.path.join(dataDir, 'BSE_System_Parameters.csv'), skiprows=2)
SN = pd.read_csv(os.path.join(dataDir, 'BSE_Supernovae.csv'), skiprows=2)
RLOF = pd.read_csv(os.path.join(dataDir, 'BSE_RLOF.csv'), skiprows=2)

# Remove equilibriated at birth
EAB = SP.loc[SP['Equilibrated_At_Birth']==1]
SP.drop(SP.loc[SP["    SEED    "].isin(EAB["    SEED    "])].index, inplace=True)

invalidVals = SN.loc[(SN["SystemicSpeed "] == "          -nan")|(SN["SystemicSpeed "] == "          -nan")|(SN["SystemicSpeed "] == "          -nan")]
if len(invalidVals)>0:
    print("{} systems dropped".format(len(invalidVals)))
    SN.drop(invalidVals.index, inplace=True)

SN = SN.astype({"SystemicSpeed ":"float64", 
                "ComponentSpeed(SN)":"float64", 
                "ComponentSpeed(CP)":"float64",
                "SemiMajorAxis ":"float64"})

# Find BHs with other stars
index = (((SP['Stellar_Type(1)']==14)&(SP['Stellar_Type(2)']<13))|((SP['Stellar_Type(1)']<13)&(SP['Stellar_Type(2)']==14))&(SP['Merger']==0))
BH_else = SP.loc[index]

# Find only bound systems and non-mergers
BH_else_bound = BH_else.loc[(BH_else['Unbound']==0)]
BH_else_bound.reset_index(drop=True, inplace=True)

SN_BH = SN.loc[SN['    SEED    '].isin(BH_else_bound['    SEED    '])]
SN_BH.reset_index(drop=True, inplace=True)

# Find systems that underwent RLOF
BH_else_bound_RLOF = BH_else_bound.loc[BH_else_bound['    SEED    '].isin(RLOF['  SEED>MT   '])]

print('Fraction of systems with m2<20 that remain bound')
print(len(BH_else_bound)/len(BH_else))

print('Fraction of systems that form on BH, remain bound and experience RLOF and have m2<18')
print(len(BH_else_bound_RLOF.loc[BH_else_bound_RLOF["Stellar_Type(2)"]<13])/len(BH_else_bound["Stellar_Type(2)"]<13))

print('Fraction of systens that form one BH and remain bound that have m2<18')
print(len(BH_else_bound.loc[BH_else_bound[' Mass@ZAMS(2) ']<18])/len(BH_else_bound))

print('Fraction of systems that form one BH that have m2<18')
print(len(BH_else.loc[BH_else[' Mass@ZAMS(2) ']<18])/len(BH_else))

print('Fraction of all systems where m2<18')
print(len(SP.loc[SP[' Mass@ZAMS(2) ']<18])/len(SP))

plt.hist(SN_BH['SystemicSpeed '], bins = np.logspace(np.log10(min(SN_BH['SystemicSpeed '])), np.log10(max(SN_BH['SystemicSpeed '])), 25), histtype='step', density=True, cumulative=True)
plt.xscale('log')

plt.title('Cumulative distribution of systemic kicks for BH-star binaries')
plt.xlabel('Kick $kms^{-1}$')
plt.ylabel('CDF')

plt.show()

