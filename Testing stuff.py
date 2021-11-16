import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Checking if the github authentication is working
'''

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

SN_types = {"ccSN":1,
            "ecSN":2,
            "PISN":4,
            "PPISN":8,
            "USSN":16}

print(SN.keys())

print("######## Supernoave Percentages ########\n")
for i in SN_types:
    frac = SN["SN_Type(SN)"].loc[SN["SN_Type(SN)"]==SN_types[i]].count()/len(SN)
    print("{0:.2%} of SN are {1}\n".format(frac, i))


# systems undergoing a second SN
SN_dup_1 = SN.loc[SN.duplicated(subset="    SEED    ", keep = "last")]
SN_dup_2 = SN.loc[SN.duplicated(subset="    SEED    ", keep = "first")]

#########################
'''
Here I am plotting the distribution of systemic and component kick velocities
where the applied kick has been reduced to zero
'''
sys_at_0 = SN["SystemicSpeed "].loc[SN["Applied_Kick_Magnitude(SN)"] == 0]
com_at_0 = np.append(SN["ComponentSpeed(SN)"].loc[SN["Applied_Kick_Magnitude(SN)"] == 0].values, SN["ComponentSpeed(CP)"].loc[SN["Applied_Kick_Magnitude(SN)"] == 0].values) 

plt.figure()
plt.hist(sys_at_0, bins = np.logspace(np.log10(min(sys_at_0)), np.log10(max(sys_at_0)), 25), histtype = "step", label = "Systemic kicks")
plt.hist(com_at_0, bins = np.logspace(np.log10(min(com_at_0)), np.log10(max(com_at_0)), 25), histtype = "step", label = "Component kicks")
plt.xscale("log")
plt.legend(loc="best")
plt.xlabel("Velocity $[kms^{-1}]$")
plt.title("Systemic and Component kick velocity distributions where the applied kick = 0")


# BHBH systems that remain bound
BHB = SN.loc[(SN["Unbound"]==0)&(SN["Stellar_Type(SN)"]==14)&(SN["Stellar_Type(CP)"]==14)]
BHB.reset_index(drop=True, inplace=True)

# BHBH systems which become unbound
BHB_unbound = SN.loc[(SN["Unbound"]==1)&(SN["Stellar_Type(SN)"]==14)&(SN["Stellar_Type(CP)"]==14)]
BHB_unbound.reset_index(drop=True, inplace=True)

All_BHB = SN.loc[(SN["Stellar_Type(SN)"]==14)&(SN["Stellar_Type(CP)"]==14)]
All_BHB.reset_index(drop=True, inplace=True)

# Black holes that are formed in the first supernovae but don't break the binary
BH1_bound = SN.loc[(SN.duplicated(subset=["    SEED    "], keep = "last"))&(SN["Stellar_Type(SN)"]==14)]
BH1_bound.reset_index(drop=True, inplace=True)

# Neutron stars that are formed in the first supernovae but don't break the binary
NS1_bound = SN.loc[(SN.duplicated(subset=["    SEED    "], keep = "last"))&(SN["Stellar_Type(SN)"]==13)]
NS1_bound.reset_index(drop=True, inplace=True)

# Black holes formed in the first supernovae which break the binary
BH1_unbound = SN.loc[(SN["Unbound"]==1)&(SN["Stellar_Type(SN)"]==14)&(~SN["    SEED    "].isin(SN_dup_1["    SEED    "]))]
BH1_unbound.reset_index(drop=True, inplace=True)

# BHNS systems
BHNS_bound = SN.loc[(SN["Stellar_Type(SN)"]>12)&(SN["Stellar_Type(CP)"]>12)&(SN["Stellar_Type(SN)"]!=SN["Stellar_Type(CP)"])&(SN["Unbound"]==0)]
BHNS_bound.reset_index(drop=True, inplace=True)

BHNS_unbound = SN.loc[(SN["Stellar_Type(SN)"]>12)&(SN["Stellar_Type(CP)"]>12)&(SN["Stellar_Type(SN)"]!=SN["Stellar_Type(CP)"])&(SN["Unbound"]==1)]
BHNS_unbound.reset_index(drop=True, inplace=True)

# Defining a set of possible escape velocities
v_esc = np.linspace(0, 2500, 1000) # km/s

# Setting empty arrays for the fractions
frac_retained_unbound = np.zeros_like(v_esc)
frac_retained_bound = np.zeros_like(v_esc)
frac_retained_bound_BHNS = np.zeros_like(v_esc)
frac_retained_total = np.zeros_like(v_esc)

# Total number of BHs inludes those in BHBH binaries, BHNS binaries and unbound BHs from the first or second SN 
total_BH = (len(BHB)+len(BHB_unbound))*2 + len(BHNS_bound) + len(BHNS_unbound) + len(BH1_unbound)

for i in range(len(v_esc)):
    """
    For each escape velocity, calculate the fraction of BHs that are retained for each of the possible systems that BHs exist in
    important is to first check if those binaries which survive the initial SN are retained inside the cluster, if so then we don't
    need to look at these systems for the second SN as they have already left the cluster
    """
    # Ejected on first SN?
    retained_from_first = SN_dup_1.loc[SN_dup_1["SystemicSpeed "]<v_esc[i]]

    # Number of bound BHBH systems that are retained 
    retained_bound = BHB.loc[(BHB["SystemicSpeed "]<v_esc[i])&(BHB["    SEED    "].isin(retained_from_first["    SEED    "]))]

    # Number/fraction of bound BHNS systems that are retained
    retained_bound_BHNS = BHNS_bound.loc[(BHNS_bound["SystemicSpeed "]<v_esc[i])&(BHNS_bound["    SEED    "].isin(retained_from_first["    SEED    "]))]
    frac_retained_bound_BHNS[i] = (len(retained_bound_BHNS))/total_BH

    frac_retained_bound[i] = (len(retained_bound)*2 + len(retained_bound_BHNS))/total_BH # Fractional bound systems retained (BHNS & BHBH)

    # Number/fraction of BHs from unbound BHNS systems that are retained
    retained_unbound_BHNS_SN = BHNS_unbound.loc[(BHNS_unbound["Stellar_Type(SN)"]==14)&(BHNS_unbound["ComponentSpeed(SN)"]<v_esc[i])&(BHNS_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]
    retained_unbound_BHNS_CP = BHNS_unbound.loc[(BHNS_unbound["Stellar_Type(CP)"]==14)&(BHNS_unbound["ComponentSpeed(CP)"]<v_esc[i])&(BHNS_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]

    # Number/fraction of unbound BHs that are retained
    retained_unbound_0 = BH1_unbound.loc[BH1_unbound["ComponentSpeed(SN)"]<v_esc[i]]
    retained_unbound_1 = BHB_unbound.loc[BHB_unbound["ComponentSpeed(SN)"]<v_esc[i]]
    retained_unbound_2 = BHB_unbound.loc[BHB_unbound["ComponentSpeed(CP)"]<v_esc[i]]
    frac_retained_unbound[i] = (len(retained_unbound_0) + len(retained_unbound_1)+len(retained_unbound_2)+len(retained_unbound_BHNS_CP)+len(retained_unbound_BHNS_SN))/total_BH

    # Total retained BH fraction
    frac_retained_total[i] = (len(retained_unbound_0) + len(retained_unbound_1)+len(retained_unbound_2) + 2*len(retained_bound)+len(retained_bound_BHNS)+len(retained_unbound_BHNS_CP)+len(retained_unbound_BHNS_SN))/total_BH
"""
plt.semilogx(v_esc, frac_retained_total, label = "total")
plt.semilogx(v_esc, frac_retained_bound, label = "bound")
plt.semilogx(v_esc, frac_retained_unbound, label = "unbound")

plt.legend(loc = "best")

print(frac_retained_total[1])
print(frac_retained_bound[1])
print(frac_retained_unbound[1])
print(frac_retained_bound[1]+frac_retained_unbound[1])
"""


systemic_kicks = BH1_bound["SystemicSpeed "]
systemic_kicks = systemic_kicks.append(BHB["SystemicSpeed "])
systemic_kicks.reset_index(inplace = True, drop = True)


component_kicks = np.append(SN["ComponentSpeed(SN)"].values,SN["ComponentSpeed(CP)"].values)
"""component_kicks = BH1_unbound["ComponentSpeed(SN)"]
component_kicks = component_kicks.append(BHB_unbound["ComponentSpeed(SN)"])
component_kicks = component_kicks.append(BHB_unbound["ComponentSpeed(CP)"])"""
#component_kicks.reset_index(inplace = True, drop = True)

plt.hist(systemic_kicks, bins = np.logspace(np.log10(min(systemic_kicks)), np.log10(max(systemic_kicks)), 50), cumulative=False, histtype = "step", label = "Systemic kicks")
plt.hist(component_kicks, bins = np.logspace(np.log10(min(component_kicks)), np.log10(max(component_kicks)), 50), cumulative=False, histtype = "step", label = "Component kicks")


plt.legend(loc="best")
plt.xscale("log")

plt.figure()
fallback = SN["Fallback_Fraction(SN)"]

plt.hist(fallback, bins = 25, histtype="step", density = True)
plt.xlabel("fallback")
plt.ylabel("PDF")

plt.show()