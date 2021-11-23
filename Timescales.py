import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
import os 

G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
cwd = os.getcwd()

COMPAS_Results_path = "/COMPAS_Output_1%solar_metallicity"
SN = pd.read_csv(cwd + COMPAS_Results_path + "/BSE_Supernovae.csv", skiprows=2)
SP = pd.read_csv(cwd + COMPAS_Results_path + "/BSE_System_Parameters.csv", skiprows=2)

