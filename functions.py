import os; import glob; import sys
import numpy as np
from scipy.integrate import ode

# Defining constants
G = 1.908e5 # R_sol*(M_sol)^-1*km^2*s^-2 
pc2AU = 206265 # Pc -> AU
Rsol2AU = 0.00465047 # Rsol -> AU
pcMyr2kms = 1.023 # Pc/Myr -> km/s


def find_dir(Comparing=False, subdir=''):
        '''
        Finds the likely location for the petar data files to be stored
        and gives the option to autoselect them.

        Returns data directory as a string
        '''

        # Finding possible directories where data could be stored
        directories = glob.glob(subdir+'COMPAS_Output*')

        # Create a dictionary to store the available directories and index vals
        directoryList = {str(i): directories[i] for i in range(len(directories))}

        # Print the available directories
        print('Possible Directories:\n')
        for key, val in directoryList.items():
                print(key, ':', val)
        
        chooseDirectory = input("\nWhat directory is the data stored in?   ")  
        
        if Comparing:
            if chooseDirectory=='all':
                dataDirectory = np.asarray(list(directoryList.values()), dtype='str')
                return dataDirectory

            chooseDirectory = chooseDirectory.split(',')

            if all([i in directoryList.keys() for i in chooseDirectory]):
                index = np.isin(list(directoryList.keys()), chooseDirectory)
                dataDirectory = np.asarray(list(directoryList.values()), dtype='str')[index]
            else:
                print('Could not find appropriate directory selection')
                print('Quitting')
                sys.exit()
        
        elif not Comparing:
            if chooseDirectory in directoryList.keys():
                dataDirectory = directoryList[str(chooseDirectory)]

            elif os.path.exists(str(chooseDirectory)):
                dataDirectory = str(chooseDirectory)

            else:
                print('Could not find directory\n')
                print('Quitting')
                sys.exit()

        return dataDirectory

def get_Started(Comparing=False, subdir=''):
    """
    Sets up new python scripts easier by defining the
        output directories and finding any of the three layers
        of plot directories.

        By setting comparing to True you can select multiple directories
        which can be used for comparing different runs together

        Output [if Comparing=False] >>> dataDir paths

        Output [if Comparing=True] >>> dictionary of each data set in comparison
        """
    
    # Find a single dataset
    if not Comparing:
        dataDir = find_dir(subdir=subdir)
        return dataDir

    # Comparing different datasets
    elif Comparing:
        dataSets={}

        # Keep selecting datasets until user says stop
        userDone=False
        count=0
        while not userDone:
            dataDir = find_dir(Comparing=True, subdir=subdir)

            # Check if we have an array for the datadir 
            if type(dataDir) == type(np.array([])):
                for i in dataDir:
                    name = '_'.join(i.split('_')[2:])
                    
                    # Add to datasets
                    dataSets[name] = i
            else:
                name = '_'.join(i.split('_')[2:])
                
                dataSets[name] = dataDir

            print('\nCurrently selected datasets:')
            print(dataSets.keys())
            isUserFinished = input('Are these all of the datasets to compare?[y/n]')

            if isUserFinished == 'y':
                userDone = True
        
        return dataSets

def ClusterMass(vesc, rh, w0=7):
    '''
    Finds the cluster mass given a certain half-mass radii and cluster escape velocity
    from the equation vesc = fcl * (Mcl/1e6)^(1/2) * (rh/1)^(-1/2).

    Input >>> vesc = escape velocity [km/s]
              rh = cluster half-mass radius [pc]
              w0 = king cluster concentration parameter (=7 for most my models)

    Output >>> Mcl = cluster mass [Msol]
    '''
    if w0==7:
        fcl = 119.3

    return (vesc/fcl)**2 * (rh/1) * 1e6 # Msol

def tdelay(ai,ei,m1,m2):
    """
    Calculates the GW timescale for a given binary
    semi-major axis, eccentricty and masses

    Input >>> ai = Semi Major axis [Rsol]
          ei = eccentricity
          m1, m2 = primary/secondary mass [Msol]

    Output >>> tGW = merger timescale [yrs]
    """


    # Defining useful constants
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
    YrsToSec = MyrsToSec/10**6


    #----tdelay

    #-- Choose ODE integrator
    backend = 'dopri5'

    l=len(ei)
    t_merger=[]

    for i in range(l):
        a0 = ai[i]*Rsol
        m_1 = m1[i]*Msol
        m_2 = m2[i]*Msol
        e0=ei[i]

        # If the initial ecc=0 then we have analytical solution
        if e0==0:
            beta = betaWithoutMass*m_1*m_2*(m_1+m_2)
            Te = (a0**4)/(4*beta)
            t_merger.append(Te/YrsToSec)
            continue

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
        efinal = 1e-5 #-- Maximum value of e to integrate to

        solver.set_initial_value(T0, e0) #.set_f_params(r)

        sol = [] #-- Create an empty list to store the output in (here it will be the e list)

        #-- Define a function to append the output to our list
        def solout(e, T):
            sol.append([e, T/YrsToSec])
        solver.set_solout(solout)

        #-- This line actually integrates the ODE, no loop is required
        solver.integrate(efinal)

        #-- Convert list to array
        sol = np.asarray(sol, dtype=float)

        #-- Use sol to find the location

        e = sol[:, 0]
        T = np.abs(sol[:,1])

        t_max = max(np.abs(sol[:,1]))

        tm = t_max
        #print tm

        t_merger.append(tm)

    return np.asarray(t_merger)

def calcTrh(M, rh):
    '''
    Calculate the half-mass relaxation timescale for the cluster (Myrs)
    
    Input >>> M = cluster mass (Msol)
          >>> rh = cluster half-mass radius (pc)
    
    Output >>> trh = relaxation time (Myrs)
    '''
    
    #Define G 
    G = 0.00449830997959438 # pc^3 Msol^-1 Myrs^-2
    
    const = 0.138/(50*0.809) # Msol^-1
    
    return const * np.sqrt((M*rh**3)/(G))

def calcTint(M1, M2, a, Mcl, rh, trh):
    '''
    Calculates the interaction timescale for a hard encounter
    
    Input >>> M1 = mass 1 (Msol)
          >>> M2 = mass 2 (Msol)
          >>> a = semimajor axis (AU)
          >>> Mcl = cluster Mass (Msol)
          >>> rh = cluster half mass radius (pc)
          >>> trh = relaxation time (Myrs)
          
    Output >>> tint = interaction timescale (Myrs)
    '''
    
    # Convert rh to AU
    rh*=pc2AU
    
    binary = (M1*M2)/a
    cluster = rh/(Mcl**2)
    
    return 5 * binary * cluster * trh

def calcMergeFromInteractions(m1, m2, semi, tint, N):
    '''
    Calculate the affect of an interaction and see if it would 
    lead to a binary that merges. For each binary assume that the 
    binding energy increases by 40% and the eccentricity is drawn 
    from a thermal distribution averaged over 10 times
    
    Input >>> m1, m2 = primary and secondary binary mass (Msol)
          >>> Semi = semi-major axis (rsol)
          >>> tint = time for an interaction (Myrs)
          >>> N = Number of interactions to average over
          
    Output >>> avgtdelay = coalescence time averaged over 10 interactions (Myrs)
    '''
    
    # Number of binaries
    num=m1.size
    
    # Assume hard encounter increases binding energy by 40%
    new_a = semi/1.4
    
    # Empty array to store all of the Tdelays
    delayTime_all = np.zeros(num)
    
    for i in range(N):
        # Draw an eccentricity from a thermal distribution for every binary
        esq = np.random.uniform(0,1,num)
        e = np.sqrt(esq)
        
        # Find the merger time for each of the binaries in Myrs and + interaction time
        tmerge = tdelay(ai=new_a, ei=e, m1=m1, m2=m2)/1e6
        delayTime = tmerge+tint
        
        # Append to the array we have
        delayTime_all = np.vstack((delayTime_all, delayTime))
    
    # for each binary average the tdelays
    delayTime_all = delayTime_all.T
    avgtdelay = np.mean(delayTime_all, axis=1)
    
    return avgtdelay

def scaleNumHardBBH(fhard, fBBH, Mcl, mlow=0.2, mhigh=150):
    '''
    We scale the population to the number of expected hard BBHs for a given
    cluster mass using the binary fractions found from our population models
    
    Input >>> fhard = fraction of hard BBHs to total BBHs
              fBBH  = fraction of BBHs to initial Binaries
              Mcl   = Initial Mass of the cluster to scale against (Msol)
              mlow  = Lower stellar initial mass (Msol)
              mhigh = Upper stellar initial mass (Msol)
              
    Output >>> NBBHhard = Number of hard BBHs in this cluster
    '''
    # First integrate the whole mass function to find the normalisation factor 
    # based on the required cluster mass
    A = Mcl/(1/0.7 * (0.5**0.7 - mlow**0.7) - 1/0.3 * (150**(-0.3) - 0.5**(-0.3)))
    
    # Compute the number of stars >20 Msol initially in the cluster
    numBig = A/1.3 * (20**(-1.3) - 150**(-1.5))
    
    # Scale by fractions from population
    return np.ceil(fhard * fBBH * numBig)