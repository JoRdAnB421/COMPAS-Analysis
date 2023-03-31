'''
These are some results from the first trial run of the COMPAS program. 
The settings are left as default and can be seen in the Run_Details file.
Here we plot the primary and secondary BH mass distribution for those binary black hole systems that are retained in the cluster
'''

from numpy import percentile; import numpy as np
import pandas as pd; import os; import glob; import sys
import matplotlib.pyplot as plt

#plt.rcParams.update({'font.size': 1})
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
# Setting the path to the COMPAS results in question
cwd = os.getcwd()
dataDir = find_dir()
COMPAS_Results_path = os.path.join(cwd,dataDir)
DCO = pd.read_csv(os.path.join(COMPAS_Results_path, 'BSE_Double_Compact_Objects.csv'), delimiter = ",", skiprows = 2)

# Removing systems that were equilibrated at Birth
SP = pd.read_csv(os.path.join(COMPAS_Results_path, 'BSE_System_Parameters.csv'), skiprows=2)
EAB = SP.loc[SP['Equilibrated_At_Birth']==1]

total_num = len(SP.loc[SP['Equilibrated_At_Birth']==0])

DCO.drop(DCO.loc[DCO['    SEED    '].isin(EAB['    SEED    '])].index, inplace=True)


# Black holes are indexed as 14 so we need to grab only those binary BH systems
DCO_BHBH = DCO.loc[(DCO["Stellar_Type(1)"] == 14)&(DCO["Stellar_Type(2)"] == 14)]
DCO_BHBH.reset_index(drop=True, inplace=True)

print('\n{:.1%} of the intial binaries are bound BHBs\n'.format(len(DCO_BHBH)/total_num))


print(DCO_BHBH.keys())

# Set a bandwidth for the histograms
binwidth = 1

fig, ax = plt.subplots(2,1, figsize=(4, 3))

# Make sure primary mass is the most mass of the two
index = DCO_BHBH['   Mass(1)    ']>=DCO_BHBH['   Mass(2)    ']
primMass = np.append(DCO_BHBH['   Mass(1)    '].loc[index], DCO_BHBH['   Mass(2)    '].loc[~index])
secMass =  np.append(DCO_BHBH['   Mass(2)    '].loc[index], DCO_BHBH['   Mass(1)    '].loc[~index])


# Histograms of the primary and secondary mass distributions
ax[0].hist(primMass, density=True, bins=range(0, 45+binwidth, binwidth), histtype="step", label = "Primary")
ax[0].hist(secMass, density = True, bins=range(0, 45+binwidth, binwidth), linestyle = "--", histtype="step", label = "Secondary")



#plt.xlabel("BH Mass ($M_{\odot}$)")
ax[0].set_ylabel("PDF")
ax[0].set_xlabel("BH Mass ($M_{\odot}$)")

ax[0].legend(loc="upper right")

#plt.savefig(COMPAS_Results_path + r"\hist1.png")

#plt.figure()

# Plotting a posterior desnity function of the mass distributions
semiBins = np.logspace(-2, np.log10(1.5e5), 50)
ax[1].hist(DCO_BHBH["SemiMajorAxis@DCO"], density = False, cumulative=False, bins=semiBins, histtype="step")


ax[1].set_ylabel("N")
ax[1].set_xlabel("a (AU)")
ax[1].set_xscale('log')
#ax[1].set_yscale('log')
#ax[1].set_xlim(min(DCO_BHBH["SemiMajorAxis@DCO"].values)-1, max(DCO_BHBH["SemiMajorAxis@DCO"].values)+1)

fig.tight_layout()
fig.savefig(COMPAS_Results_path + r"\PDF.pdf", dpi=400)


quantiles1 = DCO_BHBH["   Mass(1)    "].quantile([0.5, 0.75, 0.99])
quantiles2 = DCO_BHBH["   Mass(2)    "].quantile([0.5, 0.75, 0.99])

for i, j in zip(quantiles1.index, quantiles2.index):
    print('\nPrimary Mass: {0:2.0%} is {1:.1f} M_sol'.format(i, quantiles1.loc[i]))
    print('Secondary Mass: {0:2.0%} is {1:.1f} M_sol\n'.format(j, quantiles2.loc[j]))

quantiles3 = DCO_BHBH["SemiMajorAxis@DCO"].quantile([0.5, 0.75, 0.99])
for k in quantiles3.index:
        print('Separation: {0:2.0%} is {1:.1f} AU\n'.format(k, quantiles3.loc[k]))
plt.show()
