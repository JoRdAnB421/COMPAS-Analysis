import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
import os

def IMF(m):
    '''
    Calculates the probability of 18<m1<m
    '''
    return -60.307/1.3*(m**-1.3 - 18**-1.3)

# Define directory and load data
dataDir = 'COMPAS_Output_solar_metallicity'
sysParams = pd.read_csv(os.path.join(dataDir, 'BSE_System_Parameters.csv'), skiprows=2)

# Remove those binaries that are equilibrated at birth
EAB = sysParams.loc[sysParams['Equilibrated_At_Birth'] == 1]
sysParams.drop(sysParams.loc[sysParams["    SEED    "].isin(EAB["    SEED    "])].index, inplace=True)

# Find binaries where both stars are BH progenitors
largeBinary = sysParams.loc[(sysParams[' Mass@ZAMS(1) ']>18)&(sysParams[' Mass@ZAMS(2) ']>18)]

# Calculate the q values
q = sysParams[' Mass@ZAMS(2) ']/sysParams[' Mass@ZAMS(1) ']
qBig = largeBinary[" Mass@ZAMS(2) "]/largeBinary[" Mass@ZAMS(1) "]

# Make function of the q = 18/M1
mass = np.linspace(18, 130, 500)
qFunction = 18/mass

# Find cumulative counts from Kroupa IMF selection of primary.
probMass = IMF(mass)

# Plot the q vs m1
plt.plot(sysParams[' Mass@ZAMS(1) '], q, 'x')
plt.plot(largeBinary[' Mass@ZAMS(1) '], qBig, '.')
plt.plot(mass, qFunction, '-g')
plt.plot(mass, probMass, '--k')

print('Fraction of systems with small secondary mass: {0:.1%}'.format(1-len(largeBinary)/len(sysParams)))

plt.show()