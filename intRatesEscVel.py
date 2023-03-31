import glob; import sys; import os
import pandas as pd
import functions
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from matplotlib.lines import Line2D


pd.options.mode.chained_assignment = None
plt.rcParams.update({'font.size': 20}) # Set a good font size

# Defining constants
G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
pc2AU = 206265 # Pc -> AU
Rsol2AU = 0.00465047 # Rsol -> AU
pcMyr2kms = 1.023 # Pc/Myr -> km/s


# Select the directory
dataDir = functions.get_Started()

# Load in the double compact objects as well as the system parameters
DCO = pd.read_csv(os.path.join(dataDir, 'BSE_Double_Compact_Objects.csv'), skiprows=2)
SP = pd.read_csv(os.path.join(dataDir, 'BSE_System_Parameters.csv'), skiprows=2)
SN = pd.read_csv(os.path.join(dataDir, 'BSE_Supernovae.csv'), skiprows=2)


# Find the equilibrated at birth and remove them from the DCOs
EAB = SP.loc[SP['Equilibrated_At_Birth']==1]
DCO.drop(DCO.loc[DCO['    SEED    '].isin(EAB['    SEED    '])].index, inplace=True)
SN.drop(SN.loc[SN['    SEED    '].isin(EAB['    SEED    '])].index, inplace=True)


# Specifically grab the BBHs
BBHMaster = DCO.loc[(DCO['Stellar_Type(1)']==14)&(DCO['Stellar_Type(2)']==14)].copy()
BBHMaster.reset_index(inplace=True, drop=True)


# For now I am dropping any binaries with a>1e6
BBHMaster = BBHMaster.loc[BBHMaster['SemiMajorAxis@DCO']<1e6]


'''
Here we find all of the possible BH systems so that we can
later find which have been retained.
'''

# Index for both SNs , only first and only last
SNDupIndex = SN.duplicated(subset='    SEED    ', keep=False)

SN1st = SN.loc[SN.duplicated(subset='    SEED    ', keep='last')]
SN2nd = SN.loc[SN.duplicated(subset='    SEED    ', keep='first')]

SN1st.reset_index(drop=True, inplace=True)
SN2nd.reset_index(drop=True, inplace=True)

# Two SNs
SNDup = SN.loc[SNDupIndex]
SNDup.reset_index(inplace=True, drop=True)

# Single SN
SNSing = SN.loc[~SNDupIndex]
SNSing.reset_index(inplace=True, drop=True)

# BH other star unbound and bound
BHSingUnbound = SNSing.loc[(SNSing['Stellar_Type(SN)']==14)&(SNSing['Unbound']==1)]
BHSingBound = SNSing.loc[(SNSing['Stellar_Type(SN)']==14)&(SNSing['Unbound']==0)]

BHSingUnbound.reset_index(inplace=True, drop=True)
BHSingBound.reset_index(inplace=True, drop=True)

# BBHs that remain bound
BBHBound = SN.loc[(SN['Stellar_Type(SN)']==14)&(SN['Stellar_Type(CP)']==14)&(SN['Unbound']==0)]
BBHBound.reset_index(inplace=True, drop=True)

# BH other SN
BHElse = SN2nd.loc[((SN2nd['Stellar_Type(SN)']==14)&(SN2nd['Stellar_Type(CP)']!=14))|((SN2nd['Stellar_Type(CP)']==14)&(SN2nd['Stellar_Type(SN)']!=14))]
BHElse.reset_index(drop=True, inplace=True)

# BBHs that are not bound
BBHUnbound = SN.loc[(SN['Stellar_Type(SN)']==14)&(SN['Stellar_Type(CP)']==14)&(SN['Unbound']==1)]
BBHUnbound.reset_index(inplace=True, drop=True)

# Set a range of esc velocities
#vesc_array = np.array([10,25,50,75,100,200,300,400,500])
vesc_array = np.logspace(1, np.log10(2e3), 50)


# List to store the interaction rate ratios
intTot=[]

k = 2 # Parameter for setting the interaction rate

# Finding the average BH mass
avgBHMass = np.mean(SN['   Mass(SN)   '].loc[SN['Stellar_Type(SN)']==14].values)

for vesc in vesc_array:
    # King cluster parameter
    W0 = 7

    # Cluster velocity dispersion
    sigmaCL = vesc/(np.sqrt(2*W0*(1+1/avgBHMass))) # km/s
    sigmaREL = np.sqrt(2/avgBHMass) * sigmaCL

    # seeds retained from first SN
    retainedInFirst = SN1st.loc[(SN1st['SystemicSpeed ']<vesc)&(SN1st['Unbound']==0)]
    retainedSN = SN1st.loc[(SN1st['ComponentSpeed(SN)']<vesc)&(SN1st['Unbound']==1)]
    retainedCP = SN1st.loc[(SN1st['ComponentSpeed(CP)']<vesc)&(SN1st['Unbound']==1)]

    # BBHbound retained after second
    index = BBHBound['    SEED    '].isin(retainedInFirst['    SEED    '])
    BBHRetain = BBHBound.loc[(BBHBound['SystemicSpeed ']<vesc)&(index)]


    # Check if there are any duplicates
    if len(BBHRetain)!=len(np.unique(BBHRetain['    SEED    '].values)):
        BBHRetain.drop_duplicates(subset='    SEED    ', inplace=True, keep='last')

    BBHRetain.reset_index(inplace=True, drop=True)    
    '''
    Finding hard binaries
    '''
    # Reduced Mass
    mu  = BBHRetain['   Mass(SN)   ']*BBHRetain['   Mass(CP)   ']/(BBHRetain['   Mass(SN)   ']+BBHRetain['   Mass(CP)   '])

    # calculate hard boundary
    ah  = (G * mu)/sigmaREL**2 # Rsol

    BBHRetain['ah'] = ah

    # Define the hard index
    hard = BBHRetain['SemiMajorAxis '] < BBHRetain['ah']

    # Define binary total mass
    Mtot = BBHRetain['   Mass(SN)   '] + BBHRetain['   Mass(CP)   ']
    BBHRetain['Mtot'] = Mtot

    '''
    Finding all of the retained singles (we only need to know the masses)
    '''
    singlesMass = np.array([])
    # BBHUnbound on second
    index = BBHUnbound['    SEED    '].isin(retainedInFirst['    SEED    '])

    singlesMass = np.append(singlesMass, BBHUnbound['   Mass(SN)   '].loc[(index)&(BBHUnbound['ComponentSpeed(SN)']<vesc)].values)
    singlesMass = np.append(singlesMass, BBHUnbound['   Mass(CP)   '].loc[(index)&(BBHUnbound['ComponentSpeed(CP)']<vesc)].values)
    singlesMass = np.append(singlesMass, retainedSN['   Mass(SN)   '].loc[retainedSN['Stellar_Type(SN)']==14].values)

    index = SN2nd['    SEED    '].isin(retainedCP['    SEED    '])
    singlesMass = np.append(singlesMass, SN2nd['   Mass(SN)   '].loc[(index)&(SN2nd['ComponentSpeed(SN)']<vesc)&(SN2nd['Stellar_Type(SN)']==14)].values)
    
    # Finding the average single BH mass
    singleMassMu = np.mean(singlesMass)

    # Finding the average of the binary total mass
    BBHMassMu = np.mean(BBHRetain['Mtot'].values)

    # Only take hard binaries
    BBHRetainHard = BBHRetain.loc[hard]
    BBHRetainHard.reset_index(inplace=True, drop=True)

    # Loop over every binary and calculate the ratio of the binary interaction rate and the single interaction rate
    gammaSTot=0
    gammaBTot=0

    for ind in BBHRetainHard.index:
        # Pick target binary
        targ = BBHRetainHard.loc[ind] # Target binary
        
        # Find the index of the target binary so it doesn't interact with itself
        index = BBHRetainHard.loc[BBHRetain['    SEED    ']==targ['    SEED    ']].index
        proj = BBHRetainHard.drop(index) # Projectile binaries
        
        # Relative vel disp for single projectiles
        sigmaRelSingle = sigmaCL * np.sqrt(1/avgBHMass + singleMassMu/(avgBHMass*targ['Mtot']))

        # Relative vel disp for BBH projectiles
        sigmaRelBBH = sigmaCL * np.sqrt(1/avgBHMass + BBHMassMu/(avgBHMass*targ['Mtot']))

        # Interaction of target binary with singles
        gammaS = np.sum(targ['SemiMajorAxis ']**2 * (1 + (G*(targ['Mtot']+singlesMass))/(k*targ['SemiMajorAxis ']*sigmaRelSingle**2)))

        # Interaction of target binary with other binaries
        gammaB = np.sum((targ['SemiMajorAxis '] + proj['SemiMajorAxis '].values)**2 * (1 + (G*(targ['Mtot']+proj['Mtot'].values))/(k*(targ['SemiMajorAxis '] + proj['SemiMajorAxis '].values)*sigmaRelBBH**2)))
        
        
        # Add these to the totals 
        gammaSTot+=gammaS
        gammaBTot+=gammaB  
        
        print(f'{ind/len(BBHRetainHard):.0%} Completed', end='\r')
    
    # Final interaction ratio
    if gammaSTot==0:
        print('No hard Binaries')
        intRatioTot = 0 

    else:
        intRatioTot = gammaBTot/gammaSTot
    
    # Append to the empty list
    intTot.append(intRatioTot)
    
    print(f'Completed {vesc} km/s')

intTot = np.asarray(intTot)

fig, ax = plt.subplots(figsize=(10,8))

ax.plot(vesc_array, intTot)
ax.axhline(1, 0, 1, color='black', linestyle='--')

ax.set_xlabel('$v_{\mathrm{esc}}$')
ax.set_ylabel('$\Gamma_{\mathrm{B}}/\Gamma_{\mathrm{S}}$')

ax.set_xscale('log')
plt.show()

fig.savefig(os.path.join(dataDir, 'Interaction ratio.pdf'), dpi=100)

data = np.vstack((vesc_array, intTot)).T

np.savetxt(os.path.join(dataDir, 'interactionData.txt'), data)
