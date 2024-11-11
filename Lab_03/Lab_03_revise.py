#!/usr/bin/env python3
'''
As set of tools and routines for solving the N-layer atomsphere energey
balance problem and perform some useful analysis

To reproduce figures in lab writte-up: Do the following

    ipython
    run Lab_03_revise.py
    question2() # Generate Fig 2
    question3() # Generate Fig 3, Fig 4
    question4() # Generate Fig 5
    question5() # Generate Fig 6

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

def question2():

    fsx = 8 
    fsy = 6
    legend_size = 15
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 18
    Title_size_axis = 18
    Ticks_size = 15

    # Single layer (N=1) and a range of emissivities
    N = 1
    Times = 20
    T_surface1 = np.zeros(Times+1)
    epsilon1 = np.zeros(Times+1)

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
    plt.title(f"Surface temperature vs. emissivity, N = {N}",size = Title_size)
    ax1.plot(epsilon1,T_surface1, label='N = 1', \
             color='blue', linewidth=line_width, marker = markerpattern, markersize = psize) # For Q2
    plt.xlabel('Emissivity', size = Title_size_axis)
    plt.ylabel('Surface temperature (K)', size = 15)
    plt.tick_params(axis='both', which='major', labelsize=Ticks_size)

    plt.grid(True)
    plt.legend(fontsize = legend_size)
    plt.savefig("Fig2.png",dpi=500)

def question3():

    fsx = 8 
    fsy = 6
    legend_size = 15
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 18
    Title_size_axis = 18
    Ticks_size = 15

    # 3-1. Single layer (N=1) and a range of emissivities
    N = 1
    Times = 20
    T_surface1 = np.zeros(Times+1)
    epsilon1 = np.zeros(Times+1)

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
    plt.title(f"Surface temperature vs. emissivity, N = {N}",size = Title_size)
    ax1.plot(epsilon1,T_surface1, label='N = 1', \
             color='blue', linewidth=line_width, marker = markerpattern, markersize = psize) # For Q2
    plt.scatter(e_288, TS, color='red', zorder = 5, label=f'Emissivity of atomsphere (T_surface = {TS})') # For Q3
    plt.text(e_288 * 1.005, TS, f'({e_288:.2f}, {TS}K)',\
             color = 'red', fontsize=legend_size, verticalalignment='top', horizontalalignment='left')
    plt.xlabel('Emissivity', size = Title_size_axis)
    plt.ylabel('Surface temperature (K)', size = 15)
    plt.tick_params(axis='both', which='major', labelsize=Ticks_size)

    plt.grid(True)
    plt.legend(fontsize = legend_size)
    plt.savefig("Fig3.png",dpi=500)

    # 3-2. A range of layers (N=1) and a constant emissivities (epsilon = 0.255)

    # 3-2-1. Find the expected Layers number
    Times = 10
    T_surface2 = np.zeros(Times)
    N_layers_32 = np.zeros(Times)
    e = 0.255
    for i in range (Times):
        N = i+1
        F = n_layer_atoms(N,e)
        T_surface2[i] = (F[0]/sigma)**0.25
        N_layers_32[i] = N

    fsx2 = 16
    fsy2 = 8
    fig2, ax2 = plt.subplots(1,2,figsize=(fsx2,fsy2))  
    ax2[0].set_title(f"Surface temperature vs. The number of layers, Emissivity = {e}",size = Title_size)
    ax2[0].plot(N_layers_32,T_surface2, label=f'Emissivity = {e}', \
                color='blue', linewidth = line_width, marker = markerpattern, markersize = psize)
    ax2[0].axhline(y = TS, color='orange', linestyle='--',linewidth = line_width)
    ax2[0].text(Times*0.9, TS*1.005, f'TS = {TS}K', fontsize=12,color = 'orange')
    ax2[0].scatter(N_layers_32[4], TS, color='red', zorder = 5, label=f'Expected layers of atomsphere (T_surface = {TS})')
    ax2[0].text(N_layers_32[4]*1.005, TS*0.995, f'({int(N_layers_32[4])}, {TS}K)',\
                color = 'red', fontsize=12, verticalalignment='top', horizontalalignment='left')
    ax2[0].set_xlabel('The number of layers', size = Title_size_axis)
    ax2[0].set_ylabel('Surface temperature (K)', size = Title_size_axis)
    ax2[0].tick_params(axis='both', which='major', labelsize=Ticks_size)

    ax2[0].grid(True)
    ax2[0].legend(fontsize = legend_size)

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
 
    ax2[1].set_title(f"Altitude (layers) vs. temperature, N = {N_expected_Earth}",size = Title_size)
    ax2[1].plot(T_altitude, Altitude, label='N = 5', \
                color='violet', linewidth=line_width, marker = markerpattern, markersize = psize)
    ax2[1].scatter(T_altitude[0], Altitude[0], color='green', zorder = 5, label=f'Surface temperature')
    ax2[1].text(T_altitude[0] * 0.95, Altitude[0]*0.975, f'({T_altitude[0]:.2f}K, {int(Altitude[0])})',\
                color = 'green', fontsize=12, verticalalignment='top', horizontalalignment='left')
    ax2[1].tick_params(axis='both', which='major', labelsize=Ticks_size)
    # print(T_altitude)

    ax2[1].set_xlabel('Temperature (K)', size = Title_size_axis)
    ax2[1].set_ylabel('Altitude', size = Title_size_axis)
    ax2[1].legend(fontsize = legend_size)

    ax2[1].grid(True)

    plt.tight_layout()
    plt.savefig("Fig4.png",dpi=500)
    plt.close('all') 

def question4():

    # Set figure parameters

    fsx = 16
    fsy = 8
    legend_size = 18
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 25
    Title_size_axis = 20
    Ticks_size = 15

    # 4. Find expected layers for Venus
    
    # albedo = 0.33
    albedo = 0.33
    Times = 40
    T_surface_Venus = np.zeros(Times)
    N_layers_Venus = np.zeros(Times)
    e_Venus = 1
    TS_Venus = 700
    S0_Venus = 2600
    for i in range (Times):
        N = i+1
        F = n_layer_atoms(N,e_Venus,S0_Venus,albedo)
        T_surface_Venus[i] = (F[0]/sigma)**0.25
        N_layers_Venus[i] = N

    fig3,ax3 = plt.subplots(1,2,figsize=(fsx,fsy)) 
    ax3[0].set_title(f"Albedo = {albedo}",size = Title_size)
    ax3[0].plot(N_layers_Venus,T_surface_Venus, label=f'Emissivity = {e_Venus}', \
                color='blue', linewidth = line_width, marker = markerpattern, markersize = psize)
    ax3[0].axhline(y = TS_Venus, color='orange', linestyle='--',linewidth = line_width)
    ax3[0].text(1, TS_Venus*1.005, f'TS = {TS_Venus}K', fontsize=legend_size, color = 'orange')
    ax3[0].scatter(N_layers_Venus[29], TS_Venus, color='red', zorder = 5, \
                   label=f'Expected layers (T_surface = {TS_Venus})')
    ax3[0].text(N_layers_Venus[29]*1.005, TS_Venus*0.995, f'({int(N_layers_Venus[29])}, {TS_Venus}K)',\
                color = 'red', fontsize=legend_size, verticalalignment='top', horizontalalignment='left')
    ax3[0].tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax3[0].set_xlim(0,45)
    ax3[0].set_ylim(250,800)
    ax3[0].set_xticks(np.arange(0,45,10))
    ax3[0].set_yticks(np.arange(300,810,100))
    ax3[0].grid(True)
    ax3[0].legend(fontsize = legend_size)

    #albedo = 0.76
    Times = 90
    T_surface_Venus = np.zeros(Times)
    N_layers_Venus = np.zeros(Times)
    albedo_Venus = 0.76
    for i in range (Times):
        N = i+1
        F = n_layer_atoms(N,e_Venus,S0_Venus,albedo_Venus)
        T_surface_Venus[i] = (F[0]/sigma)**0.25
        N_layers_Venus[i] = N

    ax3[1].set_title(f"Albedo = {albedo_Venus}",size = Title_size)
    ax3[1].plot(N_layers_Venus,T_surface_Venus, label=f'Emissivity = {e_Venus}', \
                color='blue', linewidth = line_width, marker = markerpattern, markersize = psize)
    ax3[1].axhline(y = TS_Venus, color='orange', linestyle='--',linewidth = line_width)
    ax3[1].text(1, TS_Venus*1.005, f'TS = {TS_Venus}K', fontsize=legend_size,color = 'orange')
    ax3[1].scatter(N_layers_Venus[85], TS_Venus, color='red', zorder = 5, \
                   label=f'Expected layers (T_surface = {TS_Venus})')
    ax3[1].text(N_layers_Venus[85]*1.005, TS_Venus*0.995, f'({int(N_layers_Venus[85])}, {TS_Venus}K)',\
                color = 'red', fontsize=legend_size, verticalalignment='top', horizontalalignment='left')
    ax3[1].tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax3[1].set_xlim(0,92)
    ax3[1].set_ylim(250,800)
    ax3[1].set_xticks(np.arange(0,100,20))
    ax3[1].set_yticks(np.arange(300,810,100))
    ax3[1].grid(True)
    ax3[1].legend(fontsize = legend_size)

    fig3.suptitle("Surface temperature(K) vs. The number of layers", size = Title_size)
    fig3.supxlabel("The number of layers",size = Title_size_axis)
    fig3.supylabel("Surface temperature (K)",size = Title_size_axis)

    plt.tight_layout()
    plt.savefig("Fig5",dpi=500)
    plt.close("all")

def question5():
    # Set figure parameters
    
    fsx = 16
    fsy = 8
    legend_size = 18
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 25
    Title_size_axis = 20
    Ticks_size = 15

    N = 5
    T_altitude = np.zeros([N+1])
    Altitude = np.zeros([N+1])
    e = 0.5

    F = n_layer_atoms_Nuclear_Winter(N,e)
    for i in range(6):
        if i == 0:
            T_altitude[i] = (F[i]/sigma)**0.25
        else:
            T_altitude[i] = (F[i]/(sigma*e))**0.25
        Altitude[i] = i

    # Set figure size
    fsx = 12
    fsy = 9

    # Generate temperature altitude
    fig = plt.figure(figsize=(fsx,fsy))
    ax = fig.add_subplot(111)  
    plt.title(f'Altitude (layers) vs. temperature in "nuclear winter"',size = Title_size)
    ax.plot(T_altitude, Altitude, label='N = 5', color='blue', linewidth=line_width, marker = markerpattern, markersize = psize)
    plt.scatter(T_altitude[0], Altitude[0], color='red', zorder = 5, label=f'Surface temperature of "nuclear winter"')
    plt.text(T_altitude[0] * 1.005, Altitude[0], f'({T_altitude[0]:.2f}K, {int(Altitude[0])})',color = 'red', fontsize=legend_size, verticalalignment='top', horizontalalignment='left')
    plt.xlim([250,300])
    plt.tick_params(axis='both', which='major', labelsize=Ticks_size)

    plt.xlabel('Temperature (K)', size = Title_size_axis)
    plt.ylabel('Altitude', size = Title_size_axis)

    plt.grid(True)
    plt.legend(fontsize = legend_size)
    plt.savefig("Fig6",dpi = 500)
    plt.close("all")
