#!/usr/bin/env python3
'''
As set of tools and routines for solving the Q5 Nuclear winter 
solve problem and perform some useful analysis

To reproduce figures in lab writte-up: Do the following
'''

import numpy as np
import matplotlib.pyplot as plt

# Define some useful constant here.
sigma = 5.67*(10**(-8)) # Steffan-Boltzman constant
def n_layer_atoms_Nuclear_Winter(nlayers,epsilon,S0 = 1350,albedo=0.33,debug = False):
    '''
    Solve the n-layer atomsphere problem and return temperature at each layer

    Parameters
    ----------
    nlayers: int
    The number of layers
    epsilon: float
    The emissivity of each layer
    S0: float
    Solar radiation fluxes, unit is W/m^2. For Earth, the default value is 1350W/m^2
    albedo: float
    The albedo of the planetary surface. For Earth, the default value is 0.33
    debug: bool
    A kind of test which could help us to testify if we have some problem in our model or not
    When debug = True, it will output elements in matrix A.
    A: float matrix
    Coefficient matrix for energy balance model
    b: float list
    The balance condition for each layer
    Ainv: float matrix
    The inverse matrix of A, for solving equation
    ----------

    Return
    ----------
    fluxes: float vector
    A vector include the radiation flux of surface and each layer
    '''
    #Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)
    b[nlayers] = -S0/4
    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i == j:
                if i == 0:
                    A[i,j] = -1
                else:
                    A[i,j] = -2
                
            else:
                if i == 0:
                    m = np.abs(i-j)-1
                    A[i,j] = (1-epsilon)**m
                else:
                    m = np.abs(i-j)-1
                    A[i,j] = epsilon*((1-epsilon)**m)

    Ainv = np.linalg.inv(A)
    fluxes = np.matmul(Ainv, b)

    if debug:
        for i in range(nlayers+1):
            for j in range(nlayers+1):
                print(f'A[i={i},j={j}] = {A[i, j]}')

    return fluxes

N = 5
T_altitude = np.zeros([N+1])
Altitude = np.zeros([N+1])
e = 0.5
psize = 6
line_width = 1
markerpattern = 'o'
F = n_layer_atoms_Nuclear_Winter(N,e)
for i in range(6):
    if i == 0:
        T_altitude[i] = (F[i]/sigma)**0.25
    else:
        T_altitude[i] = (F[i]/(sigma*e))**0.25
    Altitude[i] = i

# Set figure size
fsx = 8 
fsy = 6

fig = plt.figure(figsize=(fsx,fsy))
ax = fig.add_subplot(111)  
plt.title(f'Altitude (layers) vs. temperature in "nuclear winter", N = {N}',size = 15)
ax.plot(T_altitude, Altitude, label='N = 5', color='blue', linewidth=line_width, marker = markerpattern, markersize = psize)
plt.scatter(T_altitude[0], Altitude[0], color='red', zorder = 5, label=f'Surface temperature of "nuclear winter"')
plt.text(T_altitude[0] * 1.005, Altitude[0], f'({T_altitude[0]:.2f}K, {int(Altitude[0])})',color = 'red', fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.xlim([250,300])
plt.tick_params(axis='both', which='major', labelsize=12)
print(T_altitude)

plt.xlabel('Temperature (K)', size = 15)
plt.ylabel('Altitude', size = 15)

plt.grid(True)
plt.legend()
plt.show()

