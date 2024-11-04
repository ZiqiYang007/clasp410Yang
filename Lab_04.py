import numpy as np
import matplotlib.pyplot as plt

# Q1: Define function and validation

def heatdiffusion(xmax, tmax, dx, dt, c2=1, neumann=False, debug=True):

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

    if dt<=dx**2/c2:

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
        # Set Neumann-type boundary conditions:
        if neumann:
            U[0, j+1] = U[1, j+1]
            U[-1, j+1] = U[-2, j+1]
        
        return xgrid, tgrid, U
    else:
        print('error')

from matplotlib.colors import ListedColormap

# Get solution using your solver:
xmax = 1
tmax = 0.2
dx = 0.2
dt = 0.02

x, time, heat = heatdiffusion(xmax, tmax, dx, dt)
# x = np.concatenate((x, np.array([x[-1]])))
# time = np.concatenate((time, np.array([time[-1]])))
print(heat)
# Create a figure/axes object
fig, axes = plt.subplots(1, 1)

# Create a color map and add a color bar.
map = axes.pcolor(time, x, heat, cmap='seismic', vmin=0, vmax=1, shading='nearest')

plt.title(r"Heat diffusion model ($c^{2}$ = 1, $\Delta x$ = 0.2, $\Delta t$ = 0.02)",fontsize = 20)
plt.xlabel("Time (s)", fontsize = 15)
plt.ylabel("Depth (m)", fontsize = 15)
plt.tick_params(axis='both', which='major', labelsize = 12)

cb=plt.colorbar(map, ax=axes, label='Temperature ($C$)')
cb.ax.tick_params(labelsize=12)
cb.set_label('Temperature ($C$)', fontsize=14)
plt.show()

