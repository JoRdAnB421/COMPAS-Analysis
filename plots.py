'''
These are some results from the first trial run of the COMPAS program. 
The settings are left as default and can be seen in the Run_Details file.
Here we plot the primary and secondary BH mass distribution for those binary black hole systems that are retained in the cluster
'''

from numpy import percentile
import pandas as pd; import os
import matplotlib.pyplot as plt

#plt.rcParams.update({'font.size': 1})

# Setting the path to the COMPAS results in question
cwd = os.getcwd()
COMPAS_Results_path = os.path.join(cwd,'COMPAS_Output_10%solar_metallicity')
DCO = pd.read_csv(os.path.join(COMPAS_Results_path, 'BSE_Double_Compact_Objects.csv'), delimiter = ",", skiprows = 2)

# Removing systems that were equilibrated at Birth
SP = pd.read_csv(os.path.join(COMPAS_Results_path, 'BSE_System_Parameters.csv'), skiprows=2)
EAB = SP.loc[SP['Equilibrated_At_Birth']==1]

total_num = len(SP.loc[SP['Equilibrated_At_Birth']==0])

DCO.drop(DCO.loc[DCO['    SEED    '].isin(EAB['    SEED    '])].index, inplace=True)


# Black holes are indexed as 14 so we need to grab only those binary BH systems
DCO_BHBH = DCO.loc[(DCO["Stellar_Type(1)"] == 14)&(DCO["Stellar_Type(2)"] == 14)]
DCO_BHBH.reindex()

print('\n{:.1%} of the intial binaries are bound BHBs\n'.format(len(DCO_BHBH)/total_num))


print(DCO_BHBH.keys())

# Set a bandwidth for the histograms
binwidth = 1

fig, ax = plt.subplots(2,1, figsize=(4,3), sharex=True)


# Histograms of the primary and secondary mass distributions
ax[0].hist(DCO_BHBH["   Mass(1)    "], density=True, bins=range(0, 45+binwidth, binwidth), histtype="step", label = "Primary")
ax[0].hist(DCO_BHBH["   Mass(2)    "], density = True, bins=range(0, 45+binwidth, binwidth), linestyle = "--", histtype="step", label = "Secondary")


#plt.xlabel("BH Mass ($M_{\odot}$)")
ax[0].set_ylabel("PDF")

ax[0].legend(loc="upper right")

#plt.savefig(COMPAS_Results_path + r"\hist1.png")

#plt.figure()

# Plotting a posterior desnity function of the mass distributions
ax[1].hist(DCO_BHBH["   Mass(1)    "], density = True, cumulative = True, bins=range(0, 45+binwidth, binwidth), histtype="step", label = "Primary")
ax[1].hist(DCO_BHBH["   Mass(2)    "], density = True, cumulative = True, bins=range(0, 45+binwidth, binwidth), linestyle = "--", histtype="step", label = "Secondary")


ax[1].set_xlabel("BH Mass ($M_{\odot}$)")
ax[1].set_ylabel("CDF")

ax[1].legend(loc="upper left")

fig.tight_layout()
fig.savefig(COMPAS_Results_path + r"\PDF.pdf", dpi=400)


quantiles1 = DCO_BHBH["   Mass(1)    "].quantile([0.5, 0.75, 0.99])
quantiles2 = DCO_BHBH["   Mass(2)    "].quantile([0.5, 0.75, 0.99])

for i, j in zip(quantiles1.index, quantiles2.index):
    print('\nPrimary Mass: {0:2.0%} is {1:.1f} M_sol'.format(i, quantiles1.loc[i]))
    print('Secondary Mass: {0:2.0%} is {1:.1f} M_sol\n'.format(j, quantiles2.loc[j]))

plt.show()