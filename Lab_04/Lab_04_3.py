import numpy as np
import matplotlib.pyplot as plt

# Q3:  investigate permafrost in Kangerlussuaq, Greenland under global warming conditions

c2 = 0.25*(10**(-6))*60*60*24 # unit: m^2/day

t_kanger = np.array([-19.7,-21.0,-17.,-8.4, 2.3, 8.4, 10.7, 8.5, 3.1,-6.0,-12.0,-16.9])

def temp_kanger(t,t_shift):
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
    t_amp = (t_kanger- t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t- np.pi/2) + t_kanger.mean() + t_shift

def heatdiffusion(xmax, tmax, dx, dt, c2, t_shift, debug=False):
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
    if (dx**2)/c2 >= dt:

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
            U[0,:] = temp_kanger(tgrid, t_shift)
        U[M-1,:] = 5

        # Solve the heat_diffusion
        for j in range(0,N-1):
            for i in range(1,M-1):
                U[i,j+1] = (1-2*r)*U[i,j] + r*(U[i-1,j] + U[i+1,j]) 
        
        return xgrid, tgrid, U
    else:
        print('error')

from matplotlib.colors import ListedColormap

# # Get solution using your solver:
years = 100
xmax = 100
tmax = 365*years
dx = 1
dt = 1

# Set indexing for the final year of results:

loc = int(-365/dt) # Final 365 days of the result.

# Create a joint temp profile plot:
lw = 2
labelsize = 15
titlesize = 20
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
for i in [0.5,1,3]:
    x, time, heat = heatdiffusion(xmax, tmax, dx, dt, c2, i)
    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)
    summer = heat[:,loc:].max(axis=1)
    ax.plot(winter, x, label=f'Winter T shift = {i}',linewidth=lw)
    ax.plot(summer, x,label=f'Summer T shift = {i}',linewidth=lw, linestyle='--')
    plt.legend(loc='lower left') 
    plt.ylim(0,100)
    plt.yticks(np.arange(0,101, 10))
    ax.invert_yaxis()
plt.title(f"Ground Temperature: Kangerlussuaq, years={years}",fontsize = titlesize)
plt.xlabel("Temperature(℃)",fontsize = labelsize)
plt.ylabel("Depth(m)",fontsize = labelsize)
plt.xlim(-8,6)
plt.xticks(np.arange(-8, 8, 2))
plt.grid(True)
plt.show()


