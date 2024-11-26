# !/usr/bin/env python3

'''
Draft code for Lab 5: SNOWBALL EARTH!!!

To reproduce figures in lab writte-up: Do the following

    ipython
    run Lab_05.py
    Q1fig3()
    Q2fig4()
    Q2fig5()
    Q2fig6()
    Q3fig7()
    Q4fig8fig9()

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib.image as mpimg

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)

# figure parameters
fsx = 12
fsy = 8
legend_size = 10
psize = 8
line_width = 5
markerpattern = 'o'
Title_size = 20
Title_size_axis = 20
Ticks_size = 20

def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitudes bins.
    
    Returns
    -------
    lats : Numpy array
        Array of cell center latitudes
    '''

    dlat = 180 / nbins # Latitude spacing
    lats = np.arange(0,180,dlat) + dlat/2

    return dlat, lats

def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation


def snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., spherecorr=True, debug=False, \
    albedo = 0.3, dynamic_albedo = False, emiss = 1, S0 = 1370, warm = True, Temp_insert = [60]*18, gamma = 1,print_albedo = False):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn on or off debug print statements.
    albedo : float, defaults to 0.3
        Set the Earth's albedo
    dynamic albedo : bool, defaults to False
        Turn on or off dynamic albedo
    emiss : float, defaults to 1.0
        Set ground emissivity. Set to zero to turn off radiative cooling.
    S0 : float, defaults to 1370
        Set incoming solar forcing constant. Chang to zero to turn off insolation.
    warm :  bool, defaults to True
        Turn on or off using warm-Earth initial condition
    Temp_insert : float array, defaults to [60]*18
        insert temperature initial condition if warm = False
    gamma : float, defaults to 1
        solar multiplier
    print_albedo : bool, defaults to False
        Turn on or off each latitude albedo print statements

    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Create initial condition:
    if warm:
        Temp = temp_warm(lats)
    else:
        Temp = np.array(Temp_insert, dtype=np.float64)

    # Set albedo:
    albedo_ice = 0.6
    albedo_gnd = 0.3

    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # S(y)
    insol = gamma*insolation(S0,lats)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    
    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)
        
        if dynamic_albedo:
            albedo = np.zeros(len(Temp))
            loc_ice = np.where(Temp<=-10)[0]
            loc_gnd = np.where(Temp>-10)[0]
            albedo[loc_ice] = albedo_ice
            albedo[loc_gnd] = albedo_gnd
        else:
            albedo = albedo

        if print_albedo and (i+1)%1000 == 0:
            print(f"This is albedo at step {i+1} of {Temp_insert[0]}:",albedo,"\n")
        
        # Apply insolation and radiative losses:
        radiative = (1-albedo)*insol - emiss*sigma*(Temp+273.15)**4
        Temp += dt_sec*radiative / (rho*C*mxdlyr)

        Temp = np.matmul(L_inv, Temp)

    return lats, Temp

def test_snowball(tstop=10000):
    '''
    Reproduce example plot in lecture/handout.

    Using our DEFAULT values (grid size, diffusion, etc.) and a warm-Earth
    initial condition, plot:
        - Initial condition
        - Plot simple diffusion only
        - Plot simple diffusion + spherical correction
        - Plot simple diff + sphere corr + insolation
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)

    # Get simple diffusion solution:
    lats, t_diff = snowball_earth(tstop = tstop, spherecorr = False, S0 = 0, emiss = 0)

    # Get diffusion + spherical correction:
    lats, t_sphe = snowball_earth(tstop = tstop, S0 = 0, emiss = 0)

    # Get diffusion + spherical + radiative correction:
    lats,t_rad = snowball_earth(tstop = tstop)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(lats, initial, label='Warm Earth Init. Cond.')
    ax.plot(lats, t_diff, label='Simple Diffusion')
    ax.plot(lats, t_sphe, label='Diffusion + Sphere. Corr.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + radiative')

    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel(r'Temperature ($^{\circ} C$)')
    ax.set_title('The temperature variation profiles with latitude')

    ax.legend(loc='best')

    fig.tight_layout()
    plt.savefig("Fig3.png", dpi = 500)

def Q1fig3():
    '''
    Produce test_snowball figure which is fig3 in the report.

    '''
    test_snowball()

def Q2fig4():
    '''
    Produce temperature profiles at equilibrium under different diffusivity and emissivity.

    Fix emissivity at 1 and vary diffusivity in 0-150, 
        plot the relationship between temperature profile and diffusivity for fig4A
    Fix diffusivity at 100 and vary emissivity in 0-1, 
        plot the relationship between temperature profile and emissivity for fig4B

    '''   
    legend_size = 20
    # change the diffusivity
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for i in np.arange(0,151,37.5):
        lats,t = snowball_earth(lam = i,emiss = 1)
        ax.plot(lats,t,label = f"diffusivity = {i}, emissivity = 1")
    
    # warm-Earth plot:
    t_warm = temp_warm(lats)
    ax.plot(lats,t_warm,label = f"warm-Earth temperature",color = "black")

    ax.set_xlabel('Latitude (0=South Pole)',size = Title_size_axis)
    ax.set_ylabel(r'Temperature ($^{\circ} C$)',size = Title_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax.set_xlim(0, 180) 
    ax.legend(loc='best', fontsize = legend_size)

    plt.tight_layout()
    plt.savefig("Fig4A.png",dpi=500)
    plt.close('all')

    # change the emissivity
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for i in np.arange(0 , 1.05, 0.25):
        lats,t = snowball_earth(lam = 100,emiss = i)
        ax.plot(lats,t,label = f"diffusivity = 100, emissivity = {i}")
    
    # warm-Earth plot:
    t_warm = temp_warm(lats)
    ax.plot(lats,t_warm,label = f"warm-Earth temperature",color = "black")

    ax.set_xlabel('Latitude (0=South Pole)',size = Title_size_axis)
    ax.set_ylabel(r'Temperature ($^{\circ} C$)',size = Title_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax.set_xlim(0, 180) 
    ax.legend(loc='best', fontsize = legend_size)

    plt.tight_layout()
    plt.savefig("Fig4B.png",dpi=500)
    plt.close('all')

    # The parameters of figure
    fsx = 16
    fsy = 6
    legend_size = 15
    fig2,axes = plt.subplots(1,2,figsize = (fsx,fsy))

    # Import subplots
    img1 = mpimg.imread(f'Fig4A.png')
    axes[0].imshow(img1)
    axes[0].axis('off')

    img2 = mpimg.imread(f'Fig4B.png')
    axes[1].imshow(img2)
    axes[1].axis('off')

    # Add the number of subplots

    axes[0].text(0.05, 0.95, "A", transform=axes[0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axes[1].text(0.05, 0.95, "B", transform=axes[1].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    # Generate the final figure and save
    fig2.suptitle('Temperture profiles under different emissivity and diffusivity',size = Title_size)
    # plt.tight_layout()
    fig2.savefig(f"Fig4.png",dpi = 500)
    plt.close("all")

def Q2fig5():
    '''
    Produce the descrepancy between temperature profiles at equilibrium under different diffusivity and emissivity 
    and warm-Earth profile.

    Fig5A: Descrepancy between different diffusivity temperature profiles and warm-Earth profile
    Fig5B: Descrepancy between different emissivity temperature profiles and warm-Earth profile

    ''' 
    legend_size = 20
    deltat_square = np.arange(0,5,1)   
    # change the diffusivity
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    index = 0
    for i in np.arange(0,151,37.5):
        lats,t = snowball_earth(lam = i,emiss = 1)
        t_warm = temp_warm(lats)
        deltat_square[index] = np.sum((t-t_warm)**2)
        index = index+1
    
    ax.plot(np.arange(0,151,37.5), np.log10(deltat_square), label = f"temperature difference with warm-Earth",color = "orange")

    ax.set_xlabel('Diffuisivity',size = Title_size_axis)
    ax.set_ylabel(r'Log(The sum of $\Delta$$T^2$ ($K^2$))',size=Title_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax.set_xlim(0, 150) 
    # ax.legend(loc='best',fontsize = legend_size)

    plt.tight_layout()
    plt.savefig("Fig5A.png",dpi=500)
    plt.close('all')

    # change the emissivity
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    index = 0
    for i in np.arange(0 , 1.05, 0.25):
        lats,t = snowball_earth(lam = 100,emiss = i)
        t_warm = temp_warm(lats)
        deltat_square[index] = np.sum((t-t_warm)**2)
        index = index+1
    
    ax.plot(np.arange(0,1.05,0.25), np.log10(deltat_square), label = f"temperature difference with warm-Earth",color = "orange")

    ax.set_xlabel('Emissivity',size = Title_size_axis)
    ax.set_ylabel(r'Log(The sum of $\Delta$$T^2$ ($K^2$))',size = Title_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax.set_xlim(0, 1.1) 
    ax.legend(loc='best',fontsize = legend_size)

    plt.tight_layout()
    plt.savefig("Fig5B.png",dpi=500)
    plt.close('all')

    # The parameters of figure
    fsx = 16
    fsy = 6
    legend_size = 20
    fig2,axes = plt.subplots(1,2,figsize = (fsx,fsy))

    # Import subplots
    img1 = mpimg.imread(f'Fig5A.png')
    axes[0].imshow(img1)
    axes[0].axis('off')

    img2 = mpimg.imread(f'Fig5B.png')
    axes[1].imshow(img2)
    axes[1].axis('off')

    # Add the number of subplots

    axes[0].text(0.05, 0.95, "A", transform=axes[0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axes[1].text(0.05, 0.95, "B", transform=axes[1].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    # Generate the final figure and save
    fig2.suptitle('Average temperature discrepancy between warm-Earth and different emissivity and diffusivity',size = Title_size)
    plt.tight_layout()
    fig2.savefig(f"Fig5.png",dpi = 500)
    plt.close("all")

def Q2fig6():

    '''
    Produce the profile with ideal diffusivity and emissivity.

    Use diffusivity = 40 and emissivity = 0.73 to generate the temperature profile, 
    which is close to the warm-Earth profile generated by temp_warm

    ''' 
    legend_size = 15
    lam = 40
    emiss = 0.73
    lats,t = snowball_earth(lam = lam,emiss = emiss)
    t_warm = temp_warm(lats)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(lats, t_warm, label = f"Warm-Earth",color = "black")
    ax.plot(lats, t, label = f"Diffusivity = {lam}, Emissivity = {emiss}",color = "orange")
    
    ax.set_xlabel('Latitude (0=South Pole)',size = Title_size_axis)
    ax.set_ylabel(r'Temperature ($^{\circ} C$)',size = Title_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax.set_xlim(0, 180) 
    ax.legend(loc='best',fontsize=legend_size)

    plt.title('Ideal Emissivity and Diffusivity',size = Title_size)    
    plt.tight_layout()
    plt.savefig("Fig6.png",dpi=500)
    plt.close('all')

def Q3fig7():
    '''
    Produce temperature profiles with different initial condition and albedo.

    Use dynamic albedo (T>-10℃, albedo = 0.3; T<=-10℃, albedo = 0.6) to generate two temperature profiles:
        1. 60℃ all over the world for initial condition
        2. -60℃ all over the world for initial condition
    Use constant albedo = 0.6 and warm-Earth initial condition to generate a temperature profile
    Use constant albedo = 0.3 and warm-Earth initial condition to generate a temperature profile
    Plot the initial warm-Earth temperature profile as well

    ''' 
    legend_size = 15
    lats_hot,t_hot = snowball_earth(lam = 40, emiss = 0.73, dynamic_albedo = True, debug = False,warm = False, print_albedo=True)
    lats_cold,t_cold = snowball_earth(lam = 40, emiss = 0.73, dynamic_albedo = True, debug = False,\
                                      warm = False,Temp_insert=[-60]*18)
    lats_ab60,t_ab60 = snowball_earth(lam = 40, emiss = 0.73,albedo = 0.6)
    lats_warm,t_warm_eq = snowball_earth(lam = 40, emiss = 0.73,dynamic_albedo = True)
    # lats_testsnow,t_testsnow = snowball_earth(lam = 40, emiss = 0.73)
    # lats_0,t_0 = snowball_earth(lam = 40, emiss = 0.73, dynamic_albedo = True, debug = False,\
    #                                   warm = False,Temp_insert=[0]*18)
    # lats_snow,t_snow = snowball_earth(lam = 40, emiss = 0.73, dynamic_albedo = True, debug = False,\
    #                                   warm = False,Temp_insert=t_testsnow)
    t_warm = temp_warm(lats_ab60)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(lats_hot,t_hot,label = f"Hot-Earth",color = "red")
    ax.plot(lats_cold,t_cold,label = f"Cold-Earth",color = "blue",linestyle = '--')
    ax.plot(lats_ab60,t_ab60,label = f"Warm-Earth, albedo = 0.6",color = "orange")
    ax.plot(lats_ab60,t_warm,label = f"Warm-Earth, initial state",color = "black")
    ax.plot(lats_warm,t_warm_eq,label = f"Warm-Earth, after equilibrium",color = "violet",linestyle = '--')
    # ax.plot(lats_0,t_0,label = r"0$^{\circ} C$-Earth",color = "gold",linestyle = '--')
    # ax.plot(lats_snow,t_snow,label = r"dynamic-Earth",color = "seagreen",linestyle = '--')

    ax.set_xlabel('Latitude (0=South Pole)',size=Title_size_axis)
    ax.set_ylabel(r'Temperature ($^{\circ} C$)',size=Title_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax.set_xlim(0, 180) 
    ax.legend(loc='best',fontsize = legend_size)

    plt.title('Temperature profile under different initial condition and albedo',size = Title_size) 
    plt.tight_layout()
    plt.savefig("Fig7.png",dpi=500)
    plt.close('all')

def Q4fig8fig9():
    '''
    Fig8:
    Produce the relationship between average global temperature and "sloar multiplier" gamma.

    Use dynamic albedo (T>-10℃, albedo = 0.3; T<=-10℃, albedo = 0.6) to generate two average global temperature functions:
        1. -60℃ all over the world for initial condition and vary gamma from 0.4 to 1.4. 
            Initial condition of each step is the equilibrium temperature of last step
        2. Last equilibrium temperature of 1 as initial condition and vary gamma from 1.4 to 0.4
    
    Fig9:
    Produce the relationship between equator temperature and "sloar multiplier" gamma.

    The way is same with the fig8 but only calculate the equator temperature of each step.

    '''
    legend_size = 15
    Temp = [-60]*18
    Temp_mean = np.zeros(21)
    Temp_equator = np.zeros(21)
    gamma = np.arange(0.4,1.41,0.05)
    k=0

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    for i in np.arange(0.4,1.41,0.05):
        lats,t = snowball_earth(lam = 40, emiss = 0.73, warm = False,Temp_insert = Temp,gamma = i,dynamic_albedo=True)
        Temp_mean[k]= np.mean(t)
        Temp_equator[k] = np.mean([t[8],t[9]])
        Temp = t
        gamma[k] = i
        k = k+1
    # print(gamma,"\n")
    # print(Temp)

    ax.plot(gamma,Temp_mean,label = "Cold-Earth initial", color = "blue")
    ax2.plot(gamma,Temp_equator,label = "Cold-Earth initial", color = "blue")

    l = 0
    for i in np.arange(1.4,0.39,-0.05):
        lats,t = snowball_earth(lam = 40, emiss = 0.73, warm = False,Temp_insert = Temp,gamma = i,dynamic_albedo=True)
        Temp_mean[l]= np.mean(t)
        Temp_equator[l] = np.mean([t[8],t[9]])
        Temp = t
        gamma[l] = i
        l = l+1
    # print(gamma,"\n")
    # print(Temp)

    ax.plot(gamma,Temp_mean,label = "Reverse cold-Earth initial", color = "red", linestyle = "--")
    ax2.plot(gamma,Temp_equator,label = "Reverse cold-Earth initial", color = "red", linestyle = "--")
    ax.axhline(y = np.mean(temp_warm(lats)), color='black', linestyle='--', label='Average global temperature of warm-Earth')
    ax2.axhline(y = 0, color='violet', linestyle='--', label=r'0 $^{\circ} C$')
    print(temp_warm(lats))
    print(np.mean(temp_warm(lats)))

    ax.set_xlabel(r'$\gamma$',size = Title_size_axis)
    ax2.set_xlabel(r'$\gamma$',size = Title_size_axis)
    ax.set_ylabel(r'Average temperature of global Earth ($^{\circ} C$)',size = Title_size_axis-2)
    ax2.set_ylabel(r'Temperature of the equator ($^{\circ} C$)',size = Title_size_axis-2)
    ax.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax2.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax.set_xlim(0.3, 1.5) 
    ax2.set_xlim(0.3,1.5)
    ax.legend(loc='best',fontsize = legend_size)
    ax2.legend(loc='best',fontsize = legend_size)

    fig.suptitle(r'Average global temperature under different $\gamma$',size = Title_size)
    fig2.suptitle(r'The temperature of equator under different $\gamma$',size = Title_size)
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig("Fig8.png",dpi=500)
    fig2.savefig("Fig9.png",dpi=500)
    plt.close('all')
