'''
These are some results from the first trial run of the COMPAS program. 
The settings are left as default and can be seen in the Run_Details file.
Here we plot the primary and secondary BH mass distribution for those binary black hole systems that are retained in the cluster
'''

import pandas as pd
import matplotlib.pyplot as plt

# Setting the path to the COMPAS results in question
COMPAS_Results_path = r"C:\Users\jorda\OneDrive\Desktop\PhD\COMPAS Results\COMPAS_Output_solar_metallicity"
df = pd.read_csv(COMPAS_Results_path + r"\BSE_Double_Compact_Objects.csv", delimiter = ",", skiprows = 2)

# Black holes are indexed as 14 so we need to grab only those binary BH systems
df_BHBH = df.loc[(df["Stellar_Type(1)"] == 14)&(df["Stellar_Type(2)"] == 14)]
df_BHBH.reindex()

print(df_BHBH.keys())

# Set a bandwidth for the histograms
binwidth = 1

plt.figure()

# Histograms of the primary and secondary mass distributions
plt.hist(df_BHBH["   Mass(1)    "], bins=range(0, 50+binwidth, binwidth), histtype="step", label = "Primary")
plt.hist(df_BHBH["   Mass(2)    "], bins=range(0, 50+binwidth, binwidth), linestyle = "--", histtype="step", label = "Secondary")


plt.xlabel("BH Mass ($M_{\odot}$)")
plt.ylabel("Hist")

plt.legend(loc="upper left")

plt.savefig(COMPAS_Results_path + r"\hist1.png")

plt.figure()

# Plotting a posterior desnity function of the mass distributions
plt.hist(df_BHBH["   Mass(1)    "], density = True, bins=range(0, 50+binwidth, binwidth), histtype="step", label = "Primary")
plt.hist(df_BHBH["   Mass(2)    "], density = True, bins=range(0, 50+binwidth, binwidth), linestyle = "--", histtype="step", label = "Secondary")


plt.xlabel("BH Mass ($M_{\odot}$)")
plt.ylabel("PDF")

plt.legend(loc="upper right")

plt.savefig(COMPAS_Results_path + r"\PDF.png")

plt.show()