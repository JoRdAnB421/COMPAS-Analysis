import glob; import os; import sys
import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

pd.options.mode.chained_assignment = None  

# Constants and conversions
# Gravitational constant
G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
Rsol = 6.9*(10**8.) #in meters
MyrsToSec = 3.15*(10**13.) #time in sec
tobs = 13*(10**3.)*MyrsToSec #Age of MilkyWay

Gsi =  6.6*10**-11. #garavitaional constant in SI
c = 3.*(10**8.) #velocity of light in seconds
AUtoRsol = 214.9 #AU to Rsol
Msol = 2.*(10**30) #Solar mass in kg
betaWithoutMass = (64./5.)*(Gsi**3.0)/(c**5.0)
daysToSeconds = 86400
GyrsToSec = MyrsToSec * 1000
YrsToSec = 3600*24*365.25
G_pcMsolyr = 4.49e-15 # Gravitational constant in pc^3/Msol/year

def find_dir():
        '''
        Finds the likely location for the petar data files to be stored
        and gives the option to autoselect them.

        Returns data directory as a string
        '''

        # Finding possible directories where data could be stored
        directories = glob.glob("COMPAS_Output*")

        # Create a dictionary to store the available directories and index vals
        directoryList = {str(i): directories[i] for i in range(len(directories))}

        # Print the available directories
        print("Possible Directories:\n")
        for key, val in directoryList.items():
                print(key, ":", val)

        # Asking what directory the data is stored in and giving a list of potential directories
        chooseDirectory = input("\nWhat directory is the data stored in?  ")
        if chooseDirectory in directoryList.keys():
                dataDirectory = directoryList[str(chooseDirectory)]

        elif os.path.exists(str(chooseDirectory)):
                dataDirectory = str(chooseDirectory)

        else:
                print("Could not find directory\n")
                print("Quitting")
                sys.exit()

        return dataDirectory

#----tdelay

#-- Choose ODE integrator
backend = 'dopri5'
def tdelay(ai,ei,m1,m2):
    
    l=len(ei)
    t_merger=np.array([])
    
    for i in range (l):
        a0 = ai[i]*Rsol
        m_1 = m1[i]*Msol
        m_2 = m2[i]*Msol
        e0=ei[i]
    
        c0Part1 = a0*(1. - e0**2.0)
        c0Part2 = (1.+(121./304.)*e0**2.)**(870./2299.)
        c0Part3 = e0**(12./19.)
        c0 = c0Part1/(c0Part2*c0Part3)
        beta = betaWithoutMass*m_1*m_2*(m_1+m_2)

        constant = (12./19.)*(c0**4.)/beta
        #print ((1. - e0**2.)**(3./2.))
    
        func = lambda e: constant*((e**(29./19.))*(1. + (121./304.)*e**2.)**(1181./2299.))/((1. - e**2.)**(3./2.))

        #-- Create ODE solver object
        solver = ode(func).set_integrator(backend)

        #-- Define initial and final parameters
        T0 = 0        #-- Initial value of T
        efinal = 1E-5 #-- Maximum value of e to integrate to

        solver.set_initial_value(T0, e0) #.set_f_params(r)

        sol = [] #-- Create an empty list to store the output in (here it will be the e list)
    
        #-- Define a function to append the output to our list
        def solout(e, T):
            sol.append([e, T/YrsToSec])
        solver.set_solout(solout)

        #-- This line actually integrates the ODE, no loop is required
        solver.integrate(efinal)
        
        #-- Convert list to array
        sol = np.array(sol, dtype=object)
  
        #-- Use sol to find the location
  
        e = sol[:, 0]
        T = np.abs(sol[:,1])

        t_max = max(np.abs(sol[:,1]))
        
        tm = t_max
        #print tm
        
        t_merger = np.append(t_merger, tm)

    return t_merger.T
# Finding current working directory
cwd = os.getcwd()
dataDir = find_dir()

# Setting the path to the COMPAS results 
COMPAS_Results_path = dataDir
SN = pd.read_csv(os.path.join(cwd,COMPAS_Results_path , "BSE_Supernovae.csv"), skiprows=2)
SP = pd.read_csv(os.path.join(cwd,COMPAS_Results_path , "BSE_System_Parameters.csv"), skiprows=2)

print(SN.keys())

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

# systems undergoing a second SN
SN_dup_1 = SN.loc[SN.duplicated(subset="    SEED    ", keep = "last")]
SN_dup_2 = SN.loc[SN.duplicated(subset="    SEED    ", keep = "first")]

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
BH1_bound = SN.loc[(SN.duplicated(subset=["    SEED    "], keep = "last"))&(SN["Stellar_Type(SN)"]==14)&(SN["Unbound"]==0)]
BH1_bound.reset_index(drop=True, inplace=True)

# Neutron stars that are formed in the first supernovae but don't break the binary
NS1_bound = SN.loc[(SN.duplicated(subset=["    SEED    "], keep = "last"))&(SN["Stellar_Type(SN)"]==13)]
NS1_bound.reset_index(drop=True, inplace=True)

# Black holes formed in the first supernovae which break the binary
BH1_unbound = SN.loc[(SN["Unbound"]==1)&(SN["Stellar_Type(SN)"]==14)&(SN["    SEED    "].isin(SN_dup_1["    SEED    "]))]
BH1_unbound.reset_index(drop=True, inplace=True)

# Black holes formed in the second supernovae of binaries already broken
Unbound_on_first = SN_dup_1.loc[SN_dup_1["Unbound"]==1]
BH2_unbound = SN_dup_2.loc[(SN_dup_2["Stellar_Type(SN)"]==14)&(SN_dup_2["    SEED    "].isin(Unbound_on_first["    SEED    "]))]
BH2_unbound.reset_index(drop=True, inplace=True)

# BHNS systems
BHNS_bound = SN.loc[(SN["Stellar_Type(SN)"]>12)&(SN["Stellar_Type(CP)"]>12)&(SN["Stellar_Type(SN)"]!=SN["Stellar_Type(CP)"])&(SN["Unbound"]==0)]
BHNS_bound.reset_index(drop=True, inplace=True)

BHNS_unbound = SN.loc[(SN["Stellar_Type(SN)"]>12)&(SN["Stellar_Type(CP)"]>12)&(SN["Stellar_Type(SN)"]!=SN["Stellar_Type(CP)"])&(SN["Unbound"]==1)]
BHNS_unbound.reset_index(drop=True, inplace=True)

# Defining a set of possible escape velocities
v_esc = np.logspace(0, np.log10(100), 10)
v_esc=[10]

fig, ax = plt.subplots(2,2, sharex='col', sharey='row')

ax[0,1].axis('off')
ax1 = ax[1,0]
ax2 = ax[0,0]
ax3 = ax[1,1]

for i in range(len(v_esc)):
    """
    For each escape velocity, calculate the fraction of BHs that are retained for each of the possible systems that BHs exist in
    important is to first check if those binaries which survive the initial SN are retained inside the cluster, if so then we don't
    need to look at these systems for the second SN as they have already left the cluster
    """
    # Ejected on first SN?
    retained_from_first = SN_dup_1.loc[(SN_dup_1["SystemicSpeed "]<v_esc[i])&(SN_dup_1["Unbound"]==0)]

    # Number of BHs bound in systems with non-black holes
    retained_bound_BH_else_1SN = BH_else_SN.loc[(BH_else_SN["SystemicSpeed "]<v_esc[i])]
    retained_bound_BH_else_2SN = BH_else_SN_dup[(BH_else_SN_dup["SystemicSpeed "]<v_esc[i])&(BH_else_SN_dup["    SEED    "].isin(retained_from_first["    SEED    "]))]
    
    # Number of bound BHBH systems that are retained 
    retained_bound = BHB.loc[(BHB["SystemicSpeed "]<v_esc[i])&(BHB["    SEED    "].isin(retained_from_first["    SEED    "]))]
    merger_time = tdelay(retained_bound["SemiMajorAxis "].values, retained_bound[" Eccentricity "].values, retained_bound["   Mass(SN)   "].values, retained_bound["   Mass(CP)   "].values)
    mergerInHubbleTime = merger_time[merger_time<13.7e9]

    # Unretained bound
    unretained_bound = BHB.loc[(~BHB["    SEED    "].isin(retained_bound["    SEED    "]))]
    merger_time_esc = tdelay(unretained_bound["SemiMajorAxis "].values, unretained_bound[" Eccentricity "].values, unretained_bound["   Mass(SN)   "].values, unretained_bound["   Mass(CP)   "].values)
    mergerInHubbleTime_esc = merger_time_esc[merger_time_esc<13.7e9]

    # Number/fraction of BHs from unbound BHNS systems that are retained
    retained_unbound_BHNS_SN = BHNS_unbound.loc[(BHNS_unbound["Stellar_Type(SN)"]==14)&(BHNS_unbound["ComponentSpeed(SN)"]<v_esc[i])&(BHNS_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]
    retained_unbound_BHNS_CP = BHNS_unbound.loc[(BHNS_unbound["Stellar_Type(CP)"]==14)&(BHNS_unbound["ComponentSpeed(CP)"]<v_esc[i])&(BHNS_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]

    # Number/fraction of unbound BHs that are retained
    retained_unbound_0 = BH1_unbound.loc[BH1_unbound["ComponentSpeed(SN)"]<v_esc[i]]
    retained_unbound_1 = BHB_unbound.loc[(BHB_unbound["ComponentSpeed(SN)"]<v_esc[i])&(BHB_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]
    retained_unbound_2 = BHB_unbound.loc[(BHB_unbound["ComponentSpeed(CP)"]<v_esc[i])&(BHB_unbound["    SEED    "].isin(retained_from_first["    SEED    "]))]

    retained_unbound_from_first = Unbound_on_first.loc[Unbound_on_first["ComponentSpeed(CP)"]<v_esc[i]]
    retained_unbound_3 = BH2_unbound.loc[(BH2_unbound["ComponentSpeed(SN)"]<v_esc[i])&(BH2_unbound["    SEED    "].isin(retained_unbound_from_first["    SEED    "]))]

    ax1.scatter(1-unretained_bound[" Eccentricity "], unretained_bound["SemiMajorAxis "]/AUtoRsol, marker='.', label ='v$_{{esc}}$ = {:.3g}'.format(v_esc[i]))
    ax2.hist(1-unretained_bound[" Eccentricity "], bins = np.logspace(np.log10(0.01*0.9), np.log10(1.1), 50), histtype='step', density=True, cumulative=True)
    ax3.hist(unretained_bound["SemiMajorAxis "]/AUtoRsol, bins = np.logspace(np.log10(min(unretained_bound["SemiMajorAxis "]/AUtoRsol)), np.log10(max(unretained_bound["SemiMajorAxis "]/AUtoRsol)), 50), histtype='step', density=False, cumulative=False, orientation='horizontal')


    print('{:2.1%} Completed'.format((i+1)/len(v_esc)), end='\r',)


ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlim(0.01*0.9, 1.1); ax1.set_ylim(1e-2, 1e1)
ax1.set_xlabel('1-e')
ax1.set_ylabel('a (AU)')
#fig.legend(loc='upper right')
fig.tight_layout()
plt.show()