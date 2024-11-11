#!/usr/bin/env python3
'''
As set of tools and routines for solving the N-layer atomsphere energey
balance problem and perform some useful analysis

To reproduce figures in lab writte-up: Do the following

    ipython
    run Lab_04_revise.py
    question1_fig1()
    question2_fig2()
    question2_fig3()
    question2_fig4()
    question3_fig5()
    Fig6()

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Q1: Define function and validation

def heatdiffusion(xmax, tmax, dx, dt, c2=1, condition = False, debug=True):

    '''
    Solve the heat diffusion model for certain parameters, return depth array, time array, and temperature matrix
    ----------

    Parameters
    ----------
    xmax: float
    the maxium depth of the region we study, unit is meter(m)
    
    tmax: float
    the maxium time of the region we study, unit is second(s)

    dx: float
    depth intervals for solving the numerical solution of the heat diffusion model, unit is meter(m)

    dt: float
    time intervals for solving the numerical solution of the heat diffusion model, unit is second(s)

    c2: float, default is 1
    the diffusion coefficient, unit is m^2/s

    debug: bool
    A kind of test which could help us to testify if we have some problem in our model or not
    When debug = True, the extra output will include: 
    depth and time range in our function(xmax and tmax), depth and time intervals (dx and dt),
    the heat diffusion grid dimension(M,N), 
    depth array include each point of depth(xgrid), time array include each point of depth(tgrid).

    ----------

    Return
    ----------
    xgrid: float array
    A array include a series of depth value with the interval of dx(from 0 to xmax), unit is meter(m)

    tgrid: float array
    A array include a series of time value with the interval of dt(from 0 to tmax), unit is second(s)

    U: float MxN matrix
    Each point in the matrix represent of the temperature of the ceratin depth and time, 
    the solution of the heat diffusion model, unit is ℃
    '''

    if (dt<=dx**2/c2) or (condition):

        xgrid, tgrid = np.arange(0,xmax+dx,dx),np.arange(0,tmax+dt,dt)
        M = int(np.round(xmax/dx+1))
        N = int(np.round(tmax/dt+1))

        if debug:
            print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
            print(f'Our spatial step is {dx} and time step is {dt}')
            print(f'There are {M} points in space and {N} points in time.')
            print('Here is our spatial grid:')
            print(xgrid)
            print('Here is our time grid:')
            print(tgrid)

        # Set important parameter r

        r = c2*dt/dx**2

        # U matrix

        U = np.zeros((M,N))

        # Set initial condition

        U[:,0] = 4*xgrid-4*xgrid**2

        # Set boundary condition

        U[0,:] = 0
        U[M-1,:] = 0

        # Solve the heat_diffusion
        for j in range(0,N-1):
            for i in range(1,M-1):
                U[i,j+1] = (1-2*r)*U[i,j] + r*(U[i-1,j] + U[i+1,j]) 
        
        return xgrid, tgrid, U
    else:
        print('error')

def heatdiffusion_permafrost(xmax, tmax, dx, dt, c2, T_shift = 0,debug=True):
    '''
    Solve the heat diffusion model for certain parameters, return depth array, time array, and temperature matrix
    ----------

    Parameters
    ----------
    xmax: float
    the maxium depth of the region we study, unit is meter(m)
    
    tmax: float
    the maxium time of the region we study, unit is day

    dx: float
    depth intervals for solving the numerical solution of the heat diffusion model, unit is meter(m)

    dt: float
    time intervals for solving the numerical solution of the heat diffusion model, unit is day

    c2: float
    the diffusion coefficient, unit is m^2/day

    debug: bool
    A kind of test which could help us to testify if we have some problem in our model or not
    When debug = True, the extra output will include: 
    depth and time range in our function(xmax and tmax), depth and time intervals (dx and dt),
    the heat diffusion grid dimension(M,N), 
    depth array include each point of depth(xgrid), time array include each point of depth(tgrid).

    ----------

    Return
    ----------
    xgrid: float array
    A array include a series of depth value with the interval of dx(from 0 to xmax), unit is meter(m)

    tgrid: float array
    A array include a series of time value with the interval of dt(from 0 to tmax), unit is day

    U: float MxN matrix
    Each point in the matrix represent of the temperature of the ceratin depth and time, 
    the solution of the heat diffusion model, unit is ℃
    '''
    if (dx**2)/c2 >=dt:

        xgrid, tgrid = np.arange(0,xmax+dx,dx),np.arange(0,tmax+dt,dt)
        M = int(np.round(xmax/dx+1))
        N = int(np.round(tmax/dt+1))

        if debug:
            print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
            print(f'Our spatial step is {dx} and time step is {dt}')
            print(f'There are {M} points in space and {N} points in time.')
            print('Here is our spatial grid:')
            print(xgrid)
            print('Here is our time grid:')
            print(tgrid)

        # Set important parameter r

        r = c2*dt/dx**2

        # U matrix

        U = np.zeros((M,N))

        # Set initial condition

        U[:,0] = 0

        # Set boundary condition
        for i in range(N):
            U[0,:] = temp_kanger(tgrid,T_shift)
        U[M-1,:] = 5

        # Solve the heat_diffusion
        for j in range(0,N-1):
            for i in range(1,M-1):
                U[i,j+1] = (1-2*r)*U[i,j] + r*(U[i-1,j] + U[i+1,j]) 
        
        return xgrid, tgrid, U
    else:
        print('error')

def question1_fig1():

    # Figure parameters:
    fsx = 8 
    fsy = 6
    legend_size = 15
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 18
    Title_size_axis = 18
    Ticks_size = 15  

    # Get solution using your solver:
    xmax = 1
    tmax = 0.2
    dx = 0.2
    dt = 0.02

    x, time, heat = heatdiffusion(xmax, tmax, dx, dt)

    # print(heat)

    # Create a figure/axes object
    fig= plt.figure(figsize=(fsx,fsy))
    ax = fig.add_subplot(111)

    # Create a color map and add a color bar.
    map = ax.pcolor(time, x, heat, cmap='seismic', vmin=0, vmax=1, shading='nearest')

    plt.title(r"Heat diffusion model ($c^{2}$ = 1, $\Delta x$ = 0.2, $\Delta t$ = 0.02)",fontsize = Title_size)
    plt.xlabel("Time (s)", fontsize = Title_size_axis)
    plt.ylabel("Depth (m)", fontsize = Title_size_axis)
    plt.tick_params(axis='both', which='major', labelsize = Ticks_size)
    plt.xlim(0,0.21)
    plt.xticks(np.arange(0,0.21,0.05))

    cb = plt.colorbar(map, ax=ax, label='Temperature ($C$)')
    cb.ax.tick_params(labelsize = Ticks_size)
    cb.set_label('Temperature ($C$)', fontsize = legend_size)
    plt.savefig("Fig1.png", dpi = 500)

def temp_kanger(t,T_shift = 0):
    '''
    Solving the surface temperature in Kangerlussuaq, Greenland, 
    according to the average temperature of each month and the time interval we set.
    This will return a series of temperature represent the surface temperature in Kangerlussuaq, Greenland,
    which would be used as the upper boundary condition.
    ----------

    Parameters
    ----------
    t: float array
    A a series of time value, unit is day
    ----------

    Return
    ----------
    t_amp*np.sin(np.pi/180 * t- np.pi/2) + t_kanger.mean(): float array
    a series of temperature represent the surface temperature in Kangerlussuaq, Greenland
    '''
    t_kanger = np.array([-19.7,-21.0,-17.,-8.4, 2.3, 8.4, 10.7, 8.5, 3.1,-6.0,-12.0,-16.9])
    t_amp = (t_kanger- t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t- np.pi/2) + t_kanger.mean()+T_shift

def question2_fig2():

    c2 = 0.25*(10**(-6))*60*60*24 # unit: m2/day

    # Figure parameters:
    fsx = 20
    fsy = 8
    legend_size = 15
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 18
    Title_size_axis = 18
    Ticks_size = 15 

    # Get solution using your solver:
    years = 5
    xmax = 100
    tmax = 365*years
    dx = 1
    dt = 1

    x, time, heat = heatdiffusion_permafrost(xmax, tmax, dx, dt, c2)

    fig, axes = plt.subplots(1,2,figsize=(fsx,fsy))

    # Create a color map and add a color bar.
    map = axes[0].pcolor(time/365, x, heat, cmap='seismic', vmin=-25, vmax=25, shading='nearest')

    cbar = fig.colorbar(map, ax = axes)
    axes[0].set_xlabel("Time (years)",fontsize = Title_size_axis)
    axes[0].set_ylabel("Depth (m)",fontsize = Title_size_axis)
    axes[0].tick_params(axis='both', which='major', labelsize=Ticks_size)
    axes[0].set_yticks(np.arange(0,101, 10))
    axes[0].set_xticks(np.arange(0, 5.0001, 1))
    axes[0].invert_yaxis()
    cbar.set_label('Temperature ($C$)', fontsize=Title_size_axis)
    cbar.ax.tick_params(labelsize=Ticks_size)
    

    # Set indexing for the final year of results:

    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:

    winter = heat[:, loc:].min(axis=1)
    summer = heat[:,loc:].max(axis=1)

    # Create a temp profile plot:

    axes[1].plot(winter, x, label='Winter',color = 'blue',linewidth=line_width)
    axes[1].plot(summer, x,label='Suumer',color = 'red',linewidth=line_width, linestyle='--')
    axes[1].legend(loc='lower left') 

    axes[1].invert_yaxis()
    axes[1].set_xlabel("Temperature(℃)",fontsize = Title_size_axis)
    axes[1].set_ylabel("Depth(m)",fontsize = Title_size_axis)
    axes[1].set_xlim(-8,6)
    axes[1].set_ylim(0,100)
    axes[1].tick_params(axis='both', which='major', labelsize=Ticks_size)
    axes[1].set_yticks(np.arange(0,101, 10))
    axes[1].set_xticks(np.arange(-8, 8, 2))
    axes[1].invert_yaxis()
    axes[1].grid(True)

    fig.suptitle(f"Ground Temperature: Kangerlussuaq,years = {years}", size = Title_size)
    axes[0].set_title("Diffusion heat map",size = Title_size_axis)
    axes[1].set_title("Temperature gradient",size = Title_size_axis)

    plt.savefig("Fig2.png",dpi=500)
    plt.close("all")

def question2_fig3():
    c2 = 0.25*(10**(-6))*60*60*24 # unit: m2/day

    # Figure parameters:
    fsx = 20
    fsy = 15
    legend_size = 25
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 30
    Title_size_axis = 30
    Ticks_size = 25 

    # Get solution using your solver:
    xmax = 100
    dx = 1
    dt = 1

    fig, axes = plt.subplots(3,3,figsize=(fsx,fsy))
    code = ['A','B','C','D','E','F','G','H','I']
    k = 0
    # Create a temp profile plot:
    for i in range(3):
        for j in range(3):

            years = i*30+j*10+20
            tmax = 365*years
            x, time, heat = heatdiffusion_permafrost(xmax, tmax, dx, dt, c2)
            loc = int(-365/dt) # Final 365 days of the result.

            # Extract the min values over the final year:

            winter = heat[:, loc:].min(axis=1)
            summer = heat[:,loc:].max(axis=1)
            axes[i,j].set_title(f"years = {years}",size = Title_size)
            axes[i,j].plot(winter, x, label='Winter',color = 'blue',linewidth=line_width)
            axes[i,j].plot(summer, x,label='Suumer',color = 'red',linewidth=line_width, linestyle='--')
            axes[i,j].legend(loc='lower left', fontsize = legend_size) 

            axes[i,j].invert_yaxis()
            axes[i,j].set_xlim(-8,6)
            axes[i,j].set_ylim(0,100)
            axes[i,j].tick_params(axis='both', which='major', labelsize=Ticks_size)
            axes[i,j].set_yticks(np.arange(0,101, 20))
            axes[i,j].set_xticks(np.arange(-8, 8, 2))
            axes[i,j].invert_yaxis()
            axes[i,j].text(0.05, 0.9, code[k], transform=axes[i, j].transAxes,\
                       fontsize=legend_size, fontweight='bold', color='black',\
                       ha='center', va='center')
            axes[i,j].grid(True)
            k = k+1

    fig.suptitle(f"Ground Temperature: Kangerlussuaq", size = Title_size)
    fig.supxlabel("Temperature(℃)",size = Title_size_axis)
    fig.supylabel("Depth(m)",size = Title_size_axis)

    plt.tight_layout()
    plt.savefig("Fig3.png",dpi=500)
    plt.close("all")

def question2_fig4():
    c2 = 0.25*(10**(-6))*60*60*24 # unit: m2/day

    # Figure parameters:
    fsx = 15
    fsy = 10
    legend_size = 15
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 20
    Title_size_axis = 20
    Ticks_size = 18

    # Get solution using your solver:
    years = 100
    xmax = 100
    tmax = 365*years
    dx = 1
    dt = 1

    x, time, heat = heatdiffusion_permafrost(xmax, tmax, dx, dt, c2)

    fig, ax = plt.subplots(1,1,figsize=(fsx,fsy))

    # Create a color map and add a color bar.
    map = ax.pcolor(time/365, x, heat, cmap='seismic', vmin=-25, vmax=25, shading='nearest')

    cbar = fig.colorbar(map, ax = ax)
    ax.set_title(f"Ground Temperature: Kangerlussuaq, Greenland, years = {years}",size=Title_size)
    ax.set_xlabel("Time (years)",fontsize = Title_size_axis)
    ax.set_ylabel("Depth (m)",fontsize = Title_size_axis)
    ax.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax.set_yticks(np.arange(0,101, 10))
    ax.set_xticks(np.arange(0, 101, 20))
    ax.invert_yaxis()
    cbar.set_label('Temperature ($C$)', fontsize=Title_size_axis)
    cbar.ax.tick_params(labelsize=Ticks_size)

    plt.savefig("Fig4.png",dpi=500)
    plt.close("all")

def question3_fig5():
    c2 = 0.25*(10**(-6))*60*60*24 # unit: m2/day

    # Figure parameters:
    fsx = 15
    fsy = 10
    legend_size = 15
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 20
    Title_size_axis = 20
    Ticks_size = 18

    years = 100
    xmax = 100
    tmax = 365*years
    dx = 1
    dt = 1

    # Set indexing for the final year of results:

    loc = int(-365/dt) # Final 365 days of the result.

    # Create a joint temp profile plot:

    fig, ax = plt.subplots(1, 1, figsize=(fsx, fsy))
    for i in [0.5,1,3]:
        x, time, heat = heatdiffusion_permafrost(xmax, tmax, dx, dt, c2, i)
        # Extract the min values over the final year:
        winter = heat[:, loc:].min(axis=1)
        summer = heat[:,loc:].max(axis=1)
        ax.plot(winter, x, label=f'Winter T shift = {i}',linewidth=line_width)
        ax.plot(summer, x,label=f'Summer T shift = {i}',linewidth=line_width, linestyle='--')
        plt.legend(loc='lower left',fontsize = legend_size) 
        plt.ylim(0,100)
        plt.yticks(np.arange(0,101, 10))
        ax.invert_yaxis()
    plt.title(f"Ground Temperature: Kangerlussuaq, years={years}",fontsize = Title_size)
    plt.xlabel("Temperature(℃)",fontsize = Title_size_axis)
    plt.ylabel("Depth(m)",fontsize = Title_size_axis)
    plt.xlim(-8,6)
    plt.xticks(np.arange(-8, 8, 2))
    plt.tick_params(labelsize=Ticks_size)
    plt.grid(True)
    plt.savefig("Fig5.png",dpi=500)
    plt.close("all")

def Fig6():

    # Figure parameters:
    fsx = 8 
    fsy = 6
    legend_size = 15
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 18
    Title_size_axis = 18
    Ticks_size = 15  

    # Get solution using your solver:
    xmax = 1
    tmax = 0.2
    dx = 0.1
    dt = 0.02

    x, time, heat = heatdiffusion(xmax, tmax, dx, dt,c2=1,condition=True)

    # Create a figure/axes object
    fig = plt.figure(figsize=(fsx,fsy))
    ax = fig.add_subplot(111)

    # Create a color map and add a color bar.
    map = ax.pcolor(time, x, heat, cmap='seismic', vmin=0, vmax=1, shading='nearest')

    plt.title(r"Heat diffusion model ($c^{2}$ = 1, $\Delta x$ = 0.1, $\Delta t$ = 0.02)",fontsize = Title_size)
    plt.xlabel("Time (s)", fontsize = Title_size_axis)
    plt.ylabel("Depth (m)", fontsize = Title_size_axis)
    plt.tick_params(axis='both', which='major', labelsize = Ticks_size)
    plt.xlim(0,0.21)
    plt.xticks(np.arange(0,0.21,0.05))

    cb = plt.colorbar(map, ax=ax, label='Temperature ($C$)')
    cb.ax.tick_params(labelsize = Ticks_size)
    cb.set_label('Temperature ($C$)', fontsize = legend_size)
    plt.savefig("Fig6.png", dpi = 500)
    plt.close("all")