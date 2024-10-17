#!/usr/bin/env python3
'''
As set of tools and routines for solving the N-layer atomsphere energey
balance problem and perform some useful analysis

To reproduce figures in lab writte-up: Do the following
'''

import numpy as np
import matplotlib.pyplot as plt


# Define some useful constant here.
sigma = 5.67*(10**(-8)) # Steffan-Boltzman constant
def n_layer_atoms(nlayers,epsilon,S0 = 1350,albedo=0.33,debug = False):
    '''
    Solve the n-layer atomsphere problem and return temperature at each layer
    ----------

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
    b[0] = -S0*(1-albedo)/4
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

# Set figure size
fsx = 8 
fsy = 6

# 3-1. Single layer (N=1) and a range of emissivities
N = 1
Times = 20
T_surface1 = np.zeros(Times+1)
epsilon1 = np.zeros(Times+1)
psize = 6
line_width = 1
markerpattern = 'o'
for i in range (Times+1):
    e = i/Times
    F = n_layer_atoms(N,e)
    T_surface1[i] = (F[0]/sigma)**0.25
    epsilon1[i] = e

# Calculate the emissivity for T_surface = 288K
TS = 288
S0 = 1350
albedo = 0.33
e_288 = 2-S0*(1-albedo)/(2*sigma*(TS)**4)

# Plot the relationship between T_surface and emissivity
fig1 = plt.figure(figsize=(fsx,fsy))
ax1 = fig1.add_subplot(111)  
plt.title(f"Surface temperature vs. emissivity, N = {N}",size = 15)
ax1.plot(epsilon1,T_surface1, label='N = 1', color='blue', linewidth=line_width, marker = markerpattern, markersize = psize) # For Q2
plt.scatter(e_288, TS, color='red', zorder = 5, label=f'Emissivity of atomsphere (T_surface = {TS})') # For Q3
plt.text(e_288 * 1.005, TS, f'({e_288:.2f}, {TS}K)',color = 'red', fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.xlabel('Emissivity', size = 15)
plt.ylabel('Surface temperature (K)', size = 15)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True)
plt.legend()
plt.show()


# 3-2. A range of layers (N=1) and a constant emissivities (epsilon = 0.255)

# 3-2-1. Find the expected Layers number
Times = 10
T_surface2 = np.zeros(Times)
N_layers_32 = np.zeros(Times)
psize = 6
line_width = 1
markerpattern = 'o'
e = 0.255
for i in range (Times):
    N = i+1
    F = n_layer_atoms(N,e)
    T_surface2[i] = (F[0]/sigma)**0.25
    N_layers_32[i] = N

fig2 = plt.figure(figsize=(fsx,fsy))
ax2 = fig2.add_subplot(111)  
plt.title(f"Surface temperature vs. The number of layers, Emissivity = {e}",size = 15)
ax2.plot(N_layers_32,T_surface2, label=f'Emissivity = {e}', color='blue', linewidth = line_width, marker = markerpattern, markersize = psize)
plt.axhline(y = TS, color='orange', linestyle='--',linewidth = line_width)
plt.text(Times*0.9, TS*1.005, f'TS = {TS}K', fontsize=12,color = 'orange')
plt.scatter(N_layers_32[4], TS, color='red', zorder = 5, label=f'Expected layers of atomsphere (T_surface = {TS})')
plt.text(N_layers_32[4]*1.005, TS*0.995, f'({int(N_layers_32[4])}, {TS}K)',color = 'red', fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.xlabel('The number of layers', size = 15)
plt.ylabel('Surface temperature (K)', size = 15)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True)
plt.legend()
plt.show()

# 3-2-2. Plot altitude vs. temperature
N_expected_Earth = 5
F_322 = n_layer_atoms(N_expected_Earth,e)
T_altitude = np.zeros([N_expected_Earth+1])
Altitude = np.zeros([N_expected_Earth+1])
for i in range(N_expected_Earth+1):
    if i == 0:
        T_altitude[i] = (F_322[i]/sigma)**0.25
    else:
        T_altitude[i] = (F_322[i]/(sigma*e))**0.25
    Altitude[i] = i
fig21 = plt.figure(figsize=(fsx,fsy))
ax21 = fig21.add_subplot(111)  
plt.title(f"Altitude (layers) vs. temperature, N = {N_expected_Earth}",size = 15)
ax21.plot(T_altitude, Altitude, label='N = 5', color='blue', linewidth=line_width, marker = markerpattern, markersize = psize)
plt.scatter(T_altitude[0], Altitude[0], color='red', zorder = 5, label=f'Surface temperature')
plt.text(T_altitude[0] * 0.95, Altitude[0]*0.975, f'({T_altitude[0]:.2f}K, {int(Altitude[0])})',color = 'red', fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.tick_params(axis='both', which='major', labelsize=12)
print(T_altitude)

plt.xlabel('Temperature (K)', size = 15)
plt.ylabel('Altitude', size = 15)

plt.grid(True)
plt.legend()
plt.show()


# 4. Find expected layers for Venus
 
# albedo = 0.33
Times = 40
T_surface_Venus = np.zeros(Times)
N_layers_Venus = np.zeros(Times)
psize = 6
line_width = 1
markerpattern = 'o'
e_Venus = 1
TS_Venus = 700
S0_Venus = 2600
for i in range (Times):
    N = i+1
    F = n_layer_atoms(N,e_Venus,S0_Venus)
    T_surface_Venus[i] = (F[0]/sigma)**0.25
    N_layers_Venus[i] = N

fig3 = plt.figure(figsize=(fsx,fsy))
ax3 = fig3.add_subplot(111)  
plt.title(f"Surface temperature vs. The number of layers in Venus, Emissivity = {e_Venus}",size = 15)
ax3.plot(N_layers_Venus,T_surface_Venus, label=f'Emissivity = {e_Venus}, albedo = 0.33', color='blue', linewidth = line_width, marker = markerpattern, markersize = psize)
plt.axhline(y = TS_Venus, color='orange', linestyle='--',linewidth = line_width)
plt.text(1, TS_Venus*1.005, f'TS = {TS_Venus}K', fontsize=12,color = 'orange')
plt.scatter(N_layers_Venus[29], TS_Venus, color='red', zorder = 5, label=f'Expected layers of atomsphere (T_surface = {TS})')
plt.text(N_layers_Venus[29]*1.005, TS_Venus*0.995, f'({int(N_layers_Venus[29])}, {TS_Venus}K)',color = 'red', fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.xlabel('The number of layers', size = 15)
plt.ylabel('Surface temperature (K)', size = 15)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True)
plt.legend()
plt.show()

#albedo = 0.76
Times = 90
T_surface_Venus = np.zeros(Times)
N_layers_Venus = np.zeros(Times)
psize = 6
line_width = 1
markerpattern = 'o'
e_Venus = 1
TS_Venus = 700
S0_Venus = 2600
albedo_Venus = 0.76
for i in range (Times):
    N = i+1
    F = n_layer_atoms(N,e_Venus,S0_Venus,albedo_Venus)
    T_surface_Venus[i] = (F[0]/sigma)**0.25
    N_layers_Venus[i] = N

fig3 = plt.figure(figsize=(fsx,fsy))
ax3 = fig3.add_subplot(111)  
plt.title(f"Surface temperature vs. The number of layers in Venus, Emissivity = {e_Venus}",size = 15)
ax3.plot(N_layers_Venus,T_surface_Venus, label=f'Emissivity = {e_Venus}, albedo = {albedo_Venus}', color='blue', linewidth = line_width, marker = markerpattern, markersize = psize)
plt.axhline(y = TS_Venus, color='orange', linestyle='--',linewidth = line_width)
plt.text(1, TS_Venus*1.005, f'TS = {TS_Venus}K', fontsize=12,color = 'orange')
plt.scatter(N_layers_Venus[85], TS_Venus, color='red', zorder = 5, label=f'Expected layers of atomsphere (T_surface = {TS})')
plt.text(N_layers_Venus[85]*1.005, TS_Venus*0.995, f'({int(N_layers_Venus[85])}, {TS_Venus}K)',color = 'red', fontsize=12, verticalalignment='top', horizontalalignment='left')
plt.xlabel('The number of layers', size = 15)
plt.ylabel('Surface temperature (K)', size = 15)
plt.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True)
plt.legend()
plt.show()
