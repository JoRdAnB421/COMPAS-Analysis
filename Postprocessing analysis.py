import glob
import sys
import os
import pandas as pd
import numpy as np; from random import choices
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
pd.options.mode.chained_assignment = None  

def sq_recoil_kick(mass1, mass2, mass3, a):
    q = mass3/(mass1+mass2)
    mu12 = (mass1*mass2)/(mass1+mass2)
    M123 = (mass1+mass2+mass3)
    return 0.2*q*(G*mu12)/(a)*mass3/M123

# Gravitational constant in useful units
G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 

# Setting the different SN types
SN_types = {"ccSN":1,
            "ecSN":2,
            "PISN":4,
            "PPISN":8,
            "USSN":16}

# Finding current working directory
cwd = os.getcwd()

# Grabbing Fabio's escape velocity data from Antonini, F. and Rasio, F.A., 2016. 
data_path = "/Antonini_Rasio_data"
globular_data = np.loadtxt(cwd+data_path+"/Globular cluster hist.txt", skiprows = 4)
nuclear_data = np.loadtxt(cwd+data_path+"/Nuclear clusters.txt", skiprows = 4)

# Setting the bin edges, heights and bin widths for the histograms
glob_bin_edges = globular_data[:,0][::2]
nuc_bin_edges = nuclear_data[:,0][::2]

glob_bin_height = globular_data[:,1][1:-1:2]
glob_bin_width = np.array([glob_bin_edges[i+1] - glob_bin_edges[i] for i in range(len(glob_bin_edges)-1)])
nuc_bin_height = nuclear_data[:,1][1:-1:2]
nuc_bin_width = np.array([nuc_bin_edges[i+1] - nuc_bin_edges[i] for i in range(len(nuc_bin_edges)-1)])

# Setting the path to the COMPAS results 
#COMPAS_Results_path = r"C:\Users\jorda\OneDrive\Desktop\PhD\COMPAS Results\COMPAS_Output_solar_metallicity"
COMPAS_Results_path = "/COMPAS_Output_10%solar_metallicity"
SN = pd.read_csv((cwd+COMPAS_Results_path + "/BSE_Supernovae.csv"), skiprows=2)
SP = pd.read_csv((cwd+COMPAS_Results_path + "/BSE_System_Parameters.csv"), skiprows=2)

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

# Making a folder for the distribution plots
outdir_distributions = cwd+COMPAS_Results_path + "/Plots for different escape velocities"
if not os.path.exists(outdir_distributions): os.makedirs(outdir_distributions) 

print(SN.keys())

print("######## SUPERNOVAE PERCENTAGES ########\n")
for i in SN_types:
    # Here we find the percentage of each SN produced
    frac = SN["SN_Type(SN)"].loc[SN["SN_Type(SN)"]==SN_types[i]].count()/len(SN)
    print("{0:.2%} of SN are {1}\n".format(frac, i))


# systems undergoing a second SN
SN_dup_1 = SN.loc[SN.duplicated(subset="    SEED    ", keep = "last")]
SN_dup_2 = SN.loc[SN.duplicated(subset="    SEED    ", keep = "first")]

print("#### COMPARING FIRST & SECOND SUPERNOVAE ####\n")
for i in SN_types:
    # Here we find the percentage of each type of SN produced in the first and second event
    frac = SN["SN_Type(SN)"].loc[(SN["SN_Type(SN)"]==SN_types[i])&(~SN.index.isin(SN_dup_2.index.values))].count()/len(SN)
    print("{0:.2%} of the first SN are {1}".format(frac, i))

    frac = SN_dup_2["SN_Type(SN)"].loc[SN_dup_2["SN_Type(SN)"]==SN_types[i]].count()/len(SN)
    print("{0:.2%} of the second SN are {1}\n".format(frac, i))

print("#### COMPARING SUPERNOVAE DISRUPT THE BINARY AND THOSE THAT DON'T ####\n")
for i in SN_types:
    # Here we find the percentage of each type of SN produced in the for bound and unbound events
    frac = SN["SN_Type(SN)"].loc[(SN["SN_Type(SN)"]==SN_types[i])&(SN["Unbound"]==1)].count()/len(SN)
    print("{0:.2%} of the SN causing binary disruption are {1}".format(frac, i))

    frac = SN["SN_Type(SN)"].loc[(SN["SN_Type(SN)"]==SN_types[i])&(SN["Unbound"]==0)].count()/len(SN)
    print("{0:.2%} of the SN that do not cause binary disruption are {1}\n".format(frac, i))

# BH - other star (not NS)
index = np.where(((SP['Stellar_Type(1)']==14)&(SP['Stellar_Type(2)']<13))|((SP['Stellar_Type(1)']<13)&(SP['Stellar_Type(2)']==14))&(SP['Unbound']==0))
BH_else = SP.loc[index]

# Removing potential stellar mergers which are left over
BH_else = BH_else.loc[BH_else['Merger']==0]

# Converting to the SN file and finding the systems that experience two supernovae
BH_else_SN = SN.loc[SN["    SEED    "].isin(BH_else["    SEED    "])]

BH_else_SN_dup = BH_else_SN.loc[BH_else_SN.duplicated(subset='    SEED    ', keep=False)]
BH_else_SN.drop(BH_else_SN_dup.index, inplace=True)
BH_else_SN.reset_index(drop=True, inplace=True)
BH_else_SN_dup = BH_else_SN_dup.loc[BH_else_SN_dup.duplicated(subset='    SEED    ', keep='first')]

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
#v_esc = np.linspace(0, 2500, 1000) # km/s
v_esc = np.logspace(0, np.log10(2500), 1000)
# Setting empty arrays for the fractions
frac_retained_unbound = np.zeros_like(v_esc)
frac_retained_bound = np.zeros_like(v_esc)
frac_retained_bound_BHNS = np.zeros_like(v_esc)
frac_retained_total = np.zeros_like(v_esc)
frac_hard_bound = np.zeros_like(v_esc)
frac_hard_bound_retained_1st = np.zeros_like(v_esc)
frac_retained_bound_BH_else = np.zeros_like(v_esc)
BHB_frac = np.zeros_like(v_esc)
# Total number of BHs inludes those in BHBH binaries, BHNS binaries and unbound BHs from the first or second SN 
total_BH = (len(BHB)+len(BHB_unbound))*2 + len(BHNS_bound) + len(BHNS_unbound) + len(BH1_unbound) + len(BH_else)


for i in range(len(v_esc)):
    """
    For each escape velocity, calculate the fraction of BHs that are retained for each of the possible systems that BHs exist in
    important is to first check if those binaries which survive the initial SN are retained inside the cluster, if so then we don't
    need to look at these systems for the second SN as they have already left the cluster
    """
    # Ejected on first SN?
    retained_from_first = SN_dup_1.loc[SN_dup_1["SystemicSpeed "]<v_esc[i]]

    # Number of BHs bound in systems with non-black holes
    retained_bound_BH_else_1SN = BH_else_SN.loc[(BH_else_SN["SystemicSpeed "]<v_esc[i])]
    retained_bound_BH_else_2SN = BH_else_SN_dup[(BH_else_SN_dup["SystemicSpeed "]<v_esc[i])&(BH_else_SN_dup["    SEED    "].isin(retained_from_first["    SEED    "]))]
    
    # Number of bound BHBH systems that are retained 
    retained_bound = BHB.loc[(BHB["SystemicSpeed "]<v_esc[i])&(BHB["    SEED    "].isin(retained_from_first["    SEED    "]))]

    # Number/fraction of bound BHNS systems that are retained
    retained_bound_BHNS = BHNS_bound.loc[(BHNS_bound["SystemicSpeed "]<v_esc[i])&(BHNS_bound["    SEED    "].isin(retained_from_first["    SEED    "]))]
    frac_retained_bound_BHNS[i] = (len(retained_bound_BHNS))/total_BH

    BHB_frac[i] = 2*len(retained_bound)/(total_BH)
    frac_retained_bound[i] = (len(retained_bound)*2 + len(retained_bound_BHNS) + len(retained_bound_BH_else_1SN) + len(retained_bound_BH_else_2SN))/total_BH # Fractional bound systems retained (BHNS & BHBH)

    frac_retained_bound_BH_else[i] = (len(retained_bound_BH_else_2SN) + len(retained_bound_BHNS) + len(retained_bound_BH_else_1SN))/total_BH

    # Number/fraction of BHs from unbound BHNS systems that are retained
    retained_unbound_BHNS_SN = BHNS_unbound.loc[(BHNS_unbound["Stellar_Type(SN)"]==14)&(BHNS_unbound["ComponentSpeed(SN)"]<v_esc[i])&(BHNS_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]
    retained_unbound_BHNS_CP = BHNS_unbound.loc[(BHNS_unbound["Stellar_Type(CP)"]==14)&(BHNS_unbound["ComponentSpeed(CP)"]<v_esc[i])&(BHNS_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]

    # Number/fraction of unbound BHs that are retained
    retained_unbound_0 = BH1_unbound.loc[BH1_unbound["ComponentSpeed(SN)"]<v_esc[i]]
    retained_unbound_1 = BHB_unbound.loc[BHB_unbound["ComponentSpeed(SN)"]<v_esc[i]]
    retained_unbound_2 = BHB_unbound.loc[BHB_unbound["ComponentSpeed(CP)"]<v_esc[i]]
    frac_retained_unbound[i] = (len(retained_unbound_0) + len(retained_unbound_1)+len(retained_unbound_2)+len(retained_unbound_BHNS_CP)+len(retained_unbound_BHNS_SN))/total_BH

    # Total retained BH fraction
    frac_retained_total[i] = (len(retained_unbound_0) + len(retained_unbound_1)+len(retained_unbound_2) + 2*len(retained_bound)+len(retained_bound_BHNS)+len(retained_unbound_BHNS_CP)+len(retained_unbound_BHNS_SN)+len(retained_bound_BH_else_1SN)+len(retained_bound_BH_else_2SN))/total_BH

    # Here we find the fraction of BHBs which are considered "hard"
    sigma = v_esc[i]/4.77 # King cluster model (sigma = 1D velocity dispersion)
    
    mu = (retained_bound["   Mass(SN)   "]*retained_bound["   Mass(CP)   "])/(retained_bound["   Mass(SN)   "]+retained_bound["   Mass(CP)   "]) # M_sol

    # Hard boundary 
    ah = G*mu/sigma**2 # R_sol
    ah_a = ah/retained_bound["SemiMajorAxis "]

    hard = retained_bound.loc[ah_a>1]
    
    try:
        frac_hard_bound[i] = 2*len(hard)/(len(retained_unbound_0) + len(retained_unbound_1)+len(retained_unbound_2) + 2*len(retained_bound)+len(retained_bound_BHNS)+len(retained_unbound_BHNS_CP)+len(retained_unbound_BHNS_SN)+len(retained_bound_BH_else_1SN)+len(retained_bound_BH_else_2SN))
    except:
        frac_hard_bound[i] = 0

            # Some required quantities
    
    M12 = hard["   Mass(SN)   "] + hard["   Mass(CP)   "] # Total mass of binary M_sol
    
    # Setting up a prob distribution for the perturber mass
    lone_mass = np.append(retained_unbound_0["   Mass(SN)   "].values, retained_unbound_1["   Mass(SN)   "].values)
    lone_mass = np.append(lone_mass, retained_unbound_2["   Mass(CP)   "].values)
    lone_mass = np.append(lone_mass, retained_unbound_BHNS_CP["   Mass(CP)   "].values)
    lone_mass = np.append(lone_mass, retained_unbound_BHNS_SN["   Mass(SN)   "].values)
    
    if len(lone_mass)>1:
        try:
            values, bins = np.histogram(lone_mass, bins = range(0, round(max(lone_mass)), 1), density = True)
            bin_mid = np.array([(bins[j+1]+bins[j])/2 for j in range(len(bins)-1)])
        except:
            frac_hard_bound_retained_1st[i] = frac_retained_bound[i]
            continue

        m_perturb = choices(bin_mid, weights=values, k=len(hard))
    
    elif len(lone_mass==1):
        m_perturb = np.ones(len(hard))*lone_mass[0]
    
    else:
        frac_hard_bound_retained_1st[i] = 1
        continue

    hard["PerturbingMass"] = m_perturb

    # Setting some useful parameters
    #q3 = m3/M12
    # Fabio's suggestion is to actually set q3 = 1
    q3 = 1

    M123 = M12 + m_perturb # M_sol

    vbsq = 0.2*(G*mu)/(hard["SemiMajorAxis "])*(m_perturb/M123)
    index = np.sqrt(vbsq)<v_esc[i]
    kept_after_first = ah.loc[index]/hard["SemiMajorAxis "].loc[index]

    try:
        frac_hard_bound_retained_1st[i] = len(kept_after_first)/(len(hard))
    except:
        frac_hard_bound_retained_1st[i] = 0

fig, axes = plt.subplots(2, 1, figsize=(8,6.5), sharex=True)
#plt.tight_layout(pad=4, h_pad=-0.2)
# Plotting the fractions against the escape velocity on a loglog scale
axes[0].loglog(v_esc, frac_retained_bound_BH_else, color='tab:green', label = 'BH-else') 
axes[0].loglog(v_esc, BHB_frac, color='tab:purple', label = "Retained BHBs")
axes[0].loglog(v_esc, frac_retained_unbound, color='tab:orange', label = "Retained lone BH's")

# BHNS is now contained within bound systems
#plt.loglog(v_esc, frac_retained_bound_BHNS, label = "Bound NSBH") 

axes[0].loglog(v_esc, frac_retained_total, color='tab:blue', label = "All retained BHs")

# Fraction of those binaries that are hard

# Replicating the histogram produced in Antonini, F. and Rasio, F.A., 2016.
axes[1].bar(glob_bin_edges[:-1], height = glob_bin_height, width = glob_bin_width, fill = None, align = "edge", edgecolor = "orange", hatch = "x")
axes[1].bar(glob_bin_edges[:-1], height = glob_bin_height, width = glob_bin_width, fill = None, align = "edge", edgecolor = "gray")
axes[1].bar(nuc_bin_edges[:-1], height = nuc_bin_height, width = nuc_bin_width, fill = None, align = "edge")
fig.text(0.27, 0.4, "Globular\nclusters", color = "orange", size="large", weight = "roman")
fig.text(0.65, 0.4, "Nuclear\nclusters", size="large", weight = "roman")


#fig.text(0.5, 0.04, "$v_{esc} \ [kms^{-1}]$", ha="center", va="center")
#axes[0].set_title("[Top] Fraction of BHs (in bound binaries, alone and total) retained within different sized stellar clusters.\n[Bottom] Distribution of escape velocities from globular and nuclear clusters\n(recreation from Antonini, F. and Rasio, F.A., 2016.)")
axes[0].set_ylabel("Fraction of black holes\nretained", fontsize=15)
axes[0].legend(loc="best")
axes[0].set_xlim(1,)

axes[1].set_ylabel("PDF", fontsize=15)
axes[1].set_xlim(1,)
axes[1].set_xlabel("$v_{esc} \ [kms^{-1}]$", fontsize=15)
#plt.grid(which ="both", ls="--")
plt.tight_layout()
plt.savefig(cwd+COMPAS_Results_path + "/Fraction of black holes retained.pdf", dpi=400)

##################################################
# This is the fraction of retained stars with the points data from the first petar run
petar_data_path = r'C:\Users\c1718684\OneDrive - Cardiff University\Desktop\Petar_analysis\10000bodies'
petar_data = np.loadtxt(petar_data_path+r"\particle_fractions.txt", skiprows = 11)

plt.figure(figsize=(8,6.5))
plt.loglog(v_esc, BHB_frac/frac_retained_total, color = 'tab:purple', label = "BHB fraction")
plt.loglog(v_esc, frac_hard_bound, "--", color = 'tab:purple', label = "Hard binary fraction")
plt.loglog(v_esc, frac_hard_bound_retained_1st*frac_hard_bound, ":", color = 'tab:purple', label = "Hard binaries retained after first interaction")
plt.loglog(v_esc, (frac_retained_bound/frac_retained_total), color = 'tab:blue', label = "Binary fraction")
plt.loglog(v_esc, (frac_retained_unbound/frac_retained_total), color = 'tab:orange', label = "Singular fraction")
plt.loglog(v_esc, (frac_retained_bound_BH_else/frac_retained_total), color = 'tab:green', label="BHs with other stars")

# Plotting petar data
'''plt.scatter(petar_data[0,0], petar_data[0,1], color="tab:orange")
plt.scatter(petar_data[1,0], petar_data[1,1], color="tab:blue")
plt.scatter(petar_data[2,0], petar_data[2,1], color="tab:green")
'''
#plt.title("Fraction of retained lone BHs and BHs in binaries, normalised to the total number of\nretained BHs")
plt.ylabel("Fraction of retained blackholes", fontsize=15)
plt.xlabel("$v_{esc} \ km s^{-1}$", fontsize=15)
plt.legend(loc="best")
plt.savefig(cwd+COMPAS_Results_path + "/Fraction of retained blackholes that are in binaries.pdf", dpi=400)

################################
'''
Here I am investigating the kick velocities so that it can be somewhat compared to the escape velocities of the cluster 
'''

plt.figure(figsize = (8,6.5))
# This here would only be taking the systemic kicks for those binaries that remian bound and the component kicks from those which break the binary
systemic_kicks = np.append(BH1_bound["SystemicSpeed "].values, BHB["SystemicSpeed "].values)
component_kicks = np.append(BH1_unbound["ComponentSpeed(SN)"].values, [BHB_unbound["ComponentSpeed(SN)"].values, BHB_unbound["ComponentSpeed(CP)"].values])

# Here we just take the systemic speed and component speeds for every system, regardless of whether the binary is broken or not
'''systemic_kicks  = SN["SystemicSpeed "].values
component_kicks = np.append(SN["ComponentSpeed(SN)"].values,SN["ComponentSpeed(CP)"].values)
'''

vals_sys, bins_sys,_ = plt.hist(systemic_kicks, bins = np.logspace(np.log10(min(systemic_kicks)), np.log10(max(systemic_kicks)), 25), density = True, cumulative=True, histtype = "step", label = "Systemic kicks")
vals_com, bins_com,_ = plt.hist(component_kicks, bins = np.logspace(np.log10(min(component_kicks)), np.log10(max(component_kicks)), 25), density = True, cumulative=True, histtype = "step", label = "Component kicks")

plt.title("Cumulative distribution of both component speeds for broken binaries\nand the systemic speed of unbroken binaries")
#plt.title("Cumulative distribution all component speeds and systemic speeds")
plt.xscale("log")
plt.legend(loc="best")
plt.xlabel("$V_{kick} \ [kms^{-1}]$")
plt.ylabel("CDF")
plt.tight_layout()
plt.savefig(cwd+COMPAS_Results_path + "/Systemic and Component kick velocities.pdf", dpi=400)
'''
I want to highlight the points at 50% and 90% for each group
'''
index = np.where((abs(vals_com - 0.5)) == (min(abs(vals_com - 0.5))))
com_50 = bins_com[index]
vals_com_50 = vals_com[index]
#######################
index = np.where((abs(vals_com - 0.9)) == (min(abs(vals_com - 0.9))))
com_90 = bins_com[index]
vals_com_90 = vals_com[index]
#######################
index = np.where((abs(vals_sys - 0.5)) == (min(abs(vals_sys - 0.5))))
sys_50 = bins_sys[index]
vals_sys_50 = vals_sys[index]
#######################
index = np.where((abs(vals_sys - 0.9)) == (min(abs(vals_sys - 0.9))))
sys_90 = bins_sys[index]
vals_sys_90 = vals_sys[index]
########################
# Printing the values for the 50% velocity and 90% velocity
print("##############SYSTEMIC VELOCITIES##############\n")
print("{0:.1%} of systemic kicks are < {1:.2g} km/s\n".format(vals_sys_50[0], sys_50[0]))
print("{0:.1%} of systemic kicks are < {1:.2g} km/s\n".format(vals_sys_90[0], sys_90[0]))

print("\n##############COMPONENT VELOCITIES##############\n")
print("{0:.1%} of component kicks are < {1:.2g} km/s\n".format(vals_com_50[0], com_50[0]))
print("{0:.1%} of component kicks are < {1:.2g} km/s\n".format(vals_com_90[0], com_90[0]))

################################

'''
Here we are testing the hardening of the black holes using the method of comparing the final semi major axis a with ah,
where ah = Gmu/sigma^2 
'''

v_esc_tester = [10, 50, 100]

hardened_frac = np.zeros_like(v_esc)

binwidth = 1
linestyles = ["-", "--", ":", "-."]

# Setting a selection of different perturber masses
m3 = [5, 10, 20] # M_sol

fig, ax = plt.subplots(len(v_esc_tester), 1, figsize = (8,6.5), sharex=True)
for i in range(len(v_esc_tester)):
    '''
    This loop will find the binaries that are retained by the cluster and don't become unbound by the end of the simulation and will take the final semi-major axis of these
    binaries. 
    It will then calculate the hardening boundary via (Antonini. F & Gieles. M 2020) ah = G*mu/sigma^2, where mu = (M1*M2)/(M1+M2), sigma = velocity '''
    # Ejected on first SN?
    retained_from_first = SN_dup_1.loc[SN_dup_1["SystemicSpeed "]<v_esc_tester[i]]

    sigma = v_esc_tester[i]/4.77 # King cluster model (sigma = 1D velocity dispersion)
    
    # Number/fraction of bound BHBH systems that are retained
    retained_bound = BHB.loc[(BHB["SystemicSpeed "]<v_esc_tester[i])&(BHB["    SEED    "].isin(retained_from_first["    SEED    "]))]

    mu = (retained_bound["   Mass(SN)   "]*retained_bound["   Mass(CP)   "])/(retained_bound["   Mass(SN)   "]+retained_bound["   Mass(CP)   "]) # M_sol

    ah = G*mu/sigma**2 # R_sol
    ah_a = ah/retained_bound["SemiMajorAxis "]
    
    # Here we calculate the recoil kick from a single perturber
    # We are lookoing to see how many systems would be so hard that they would be kicked out of the cluster after a single interaction
        
    # Here we check how many of these hard binaries would be ejected after a single recoil kick
    M12 = retained_bound["   Mass(SN)   "] + retained_bound["   Mass(CP)   "] # Total mass of binary M_sol
    
    #plt.figure()
    #plt.xscale("log")
    
    vals, bin, _ = ax[i].hist(ah_a, bins = np.logspace(min(np.log10(ah_a)), max(np.log10(ah_a)), 50), density = True, cumulative = True, histtype = "step", label='$v_{{esc}}$ = {} kms$^{{-1}}$'.format(v_esc_tester[i]))
    binwidths = [bin[j+1]-bin[j] for j in range(len(bin)-1)]
    bincentres = [(bin[j+1]+bin[j])/2 for j in range(len(bin)-1)]

    for j in range(len(m3)):
        # Some required quantities
        #q3 = m3[j]/M12
        # Fabio's suggestion is to set q3=1 as it was founded out of some approximations
        q3 = 1

        M123 = M12 + m3[j] # M_sol

        vbsq = 0.2*(G*mu)/(retained_bound["SemiMajorAxis "])*(m3[j]/M123)
        index = np.sqrt(vbsq)>v_esc_tester[i]
        eject_on_first = ah.loc[index]/retained_bound["SemiMajorAxis "].loc[index]

        #plt.hist(eject_on_first, bins = np.logspace(np.log10(min(ah_a)), np.log10(max(ah_a)), 50), density = False, cumulative = False,linestyle="-.", histtype = "step", label = "m3 = {}".format(m3[j]))

    ax[i].vlines(1,0, max(vals), linestyles="--", colors="black")
    ax[i].set_xscale('log')
    #ax[i].set_yscale('log')
    ax[i].set_ylim(0.01, 1)
    #.title("ah/a distribution for binaries retained when $v_{{esc}} = {0}$".format(v_esc_tester[i]))
    ax[i].set_ylabel("CDF")
    #plt.xlabel("$a_{h}/a$")
    ax[i].legend(loc="upper left")

    #plt.savefig(outdir_distributions+"/ah_a dist for v_esc = {}.png".format(v_esc_tester[i]))
"""
    plt.figure()
    plt.xscale("log")
    plt.hist(retained_bound["SemiMajorAxis "]*0.00465, bins = np.logspace(np.log10(min(retained_bound["SemiMajorAxis "]*0.00465)), np.log10(max(retained_bound["SemiMajorAxis "]*0.00465)), 50), cumulative = False, histtype = "step", label = "ah")
    plt.title("Semi-major axis distribution for binaries retained when $v_{{esc}} = {0}$".format(v_esc_tester[i]))
    plt.ylabel("hist")
    plt.xlabel("a [$AU$]")
    
    plt.savefig(outdir_distributions+r"\a dist for v_esc = {}.png".format(v_esc_tester[i]))
"""


'''    plt.hist(retained_bound["SemiMajorAxis "].loc[retained_bound["SemiMajorAxis "] < max(ah)]*0.00465, bins = 25, density = False, cumulative=False, histtype = "step", label = "a")
    plt.title("$v_{{esc}} = {0} \ kms^{-1}$".format(v_esc_tester[i]))
    plt.ylabel("hist")
    plt.xlabel("ah [$AU$]")
    #plt.xlabel("$a_{h}/a$")
    plt.legend(loc="upper right")'''
'''
    plt.figure(10)
    plt.xscale("log")
    plt.scatter(bincentres, vals*binwidths, s = 10, label = "$v_{{esc}}$ = {}".format(v_esc_tester[i]))
    '''

'''
plt.figure(10)
plt.vlines(1,0, 0.18, linestyles="--", colors="black", label = "ah/a = 1")
plt.title("Initial mass range $M = [18 - 130] \ M_{\odot}$")
plt.xlabel("ah/a")
plt.ylabel("Fraction of systems")
plt.legend(loc="upper left")
'''

ax[-1].set_xlabel('$a_{h}/a$')
plt.tight_layout()
fig.savefig(os.path.join(outdir_distributions, 'aha distribution.pdf'), dpi=400)
plt.show()


