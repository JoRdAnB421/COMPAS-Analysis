import pandas as pd; import numpy as np
import matplotlib.pyplot as plt


COMPAS_Results_path = r"C:\Users\jorda\OneDrive\Desktop\PhD\COMPAS Results\0.01amin_30Mmin"
SN = pd.read_csv(COMPAS_Results_path + r"/BSE_Supernovae.csv", skiprows=2)
SP = pd.read_csv(COMPAS_Results_path + r"/BSE_System_Parameters.csv", skiprows=2)

SN_diff_initial_sma = {"0.01": SN,
                        "0.1": SN.loc[SN["    SEED    "].isin(SP["    SEED    "].loc[SP["SemiMajorAxis@ZAMS"]>0.1])],
                        "1"  : SN.loc[SN["    SEED    "].isin(SP["    SEED    "].loc[SP["SemiMajorAxis@ZAMS"]>1])],
                        "10" : SN.loc[SN["    SEED    "].isin(SP["    SEED    "].loc[SP["SemiMajorAxis@ZAMS"]>10])],
                        "50" : SN.loc[SN["    SEED    "].isin(SP["    SEED    "].loc[SP["SemiMajorAxis@ZAMS"]>50])],
                        "100": SN.loc[SN["    SEED    "].isin(SP["    SEED    "].loc[SP["SemiMajorAxis@ZAMS"]>100])]
                        }

#initial_sma = SN_diff_initial_sma.keys()
initial_sma = ["0.01", "0.1", "1", "10"]


# Defining a set of possible escape velocities
v_esc = np.linspace(1, 300, 1000)

colors = {"0.01": "black",
          "0.1" : "tab:blue",
          "1"   : "tab:orange",
          "10"  : "tab:green",
          "50"  : "tab:purple",
          "100" : "tab:red"
          }

legend_holder = []

plt.figure(figsize=(15,8))
for j in initial_sma:
    # systems undergoing a second SN
    SN_dup_1 = SN_diff_initial_sma[j].loc[SN_diff_initial_sma[j].duplicated(subset="    SEED    ", keep = "last")]
    
    # BHBH systems that remain bound
    BHB = SN_diff_initial_sma[j].loc[(SN_diff_initial_sma[j]["Unbound"]==0)&(SN_diff_initial_sma[j]["Stellar_Type(SN)"]==14)&(SN_diff_initial_sma[j]["Stellar_Type(CP)"]==14)]
    BHB.reset_index(drop=True, inplace=True)

    # BHBH systems which become unbound
    BHB_unbound = SN_diff_initial_sma[j].loc[(SN_diff_initial_sma[j]["Unbound"]==1)&(SN_diff_initial_sma[j]["Stellar_Type(SN)"]==14)&(SN_diff_initial_sma[j]["Stellar_Type(CP)"]==14)]
    BHB_unbound.reset_index(drop=True, inplace=True)

    All_BHB = SN_diff_initial_sma[j].loc[(SN_diff_initial_sma[j]["Stellar_Type(SN)"]==14)&(SN_diff_initial_sma[j]["Stellar_Type(CP)"]==14)]
    All_BHB.reset_index(drop=True, inplace=True)

    # Black holes that are formed in the first supernovae but don't break the binary
    BH1_bound = SN_diff_initial_sma[j].loc[(SN_diff_initial_sma[j].duplicated(subset=["    SEED    "], keep = "last"))&(SN_diff_initial_sma[j]["Stellar_Type(SN)"]==14)]
    BH1_bound.reset_index(drop=True, inplace=True)

    # Neutron stars that are formed in the first supernovae but don't break the binary
    NS1_bound = SN_diff_initial_sma[j].loc[(SN_diff_initial_sma[j].duplicated(subset=["    SEED    "], keep = "last"))&(SN_diff_initial_sma[j]["Stellar_Type(SN)"]==13)]
    NS1_bound.reset_index(drop=True, inplace=True)

    # Black holes formed in the first supernovae which break the binary
    BH1_unbound = SN_diff_initial_sma[j].loc[(SN_diff_initial_sma[j]["Unbound"]==1)&(SN_diff_initial_sma[j]["Stellar_Type(SN)"]==14)&(~SN_diff_initial_sma[j]["    SEED    "].isin(SN_dup_1["    SEED    "]))]
    BH1_unbound.reset_index(drop=True, inplace=True)

    # BHNS systems
    BHNS_bound = SN_diff_initial_sma[j].loc[(SN_diff_initial_sma[j]["Stellar_Type(SN)"]>12)&(SN_diff_initial_sma[j]["Stellar_Type(CP)"]>12)&(SN_diff_initial_sma[j]["Stellar_Type(SN)"]!=SN_diff_initial_sma[j]["Stellar_Type(CP)"])&(SN_diff_initial_sma[j]["Unbound"]==0)]
    BHNS_bound.reset_index(drop=True, inplace=True)

    BHNS_unbound = SN_diff_initial_sma[j].loc[(SN_diff_initial_sma[j]["Stellar_Type(SN)"]>12)&(SN_diff_initial_sma[j]["Stellar_Type(CP)"]>12)&(SN_diff_initial_sma[j]["Stellar_Type(SN)"]!=SN_diff_initial_sma[j]["Stellar_Type(CP)"])&(SN_diff_initial_sma[j]["Unbound"]==1)]
    BHNS_unbound.reset_index(drop=True, inplace=True)

    # Setting empty arrays for the fractions
    frac_retained_unbound = np.zeros_like(v_esc)
    frac_retained_bound = np.zeros_like(v_esc)
    frac_retained_bound_BHNS = np.zeros_like(v_esc)
    frac_retained_total = np.zeros_like(v_esc)

    total_BH = (len(BHB)+len(BHB_unbound))*2 + len(BHNS_bound) + len(BHNS_unbound) + len(BH1_unbound) 
    
    for i in range(len(v_esc)):
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
    
    plt.loglog(v_esc, frac_retained_bound, color = colors[j], linestyle = "--")
    plt.loglog(v_esc, frac_retained_unbound, color = colors[j], linestyle = "-.")
    lab, = plt.loglog(v_esc, frac_retained_total, color = colors[j], linestyle = "-")
    
    lab.set_label("$a_{min}= \ $" + j + " AU")

plt.title("Fraction of black holes retained for different minium initial semi-major axis values and constant $a_{max} = 1000 \ AU$.\n- Total BHs, -- Bound systems, -. Unbound systems")
plt.xlabel("$v_{esc} \ [kms^{-1}]$")
plt.ylabel("Fraction retained")
plt.legend(loc = "best")

plt.savefig(COMPAS_Results_path + r"/Frac retained for different amin.png")

plt.show()
