#!/usr/bin/env python3
'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do this...
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.close('all')

# 0: Died
# 1: Immune
# 2: Healthy
# 3: Sick

Times_confirm = 1 # To have significant results for same condition for recovered
Times_recovered=np.zeros((Times_confirm, 10)) # To record how many steps will it take to end up at each condition
Ratio_immune=np.zeros((Times_confirm, 10)) # To record the Final Ratio of immune/whole crowd
Ratio_fatal=np.zeros((Times_confirm, 10)) # To record the Final Ratio of fatal/whole crowd
Times_prob_spread=np.zeros((10,1)) # To record different prob_spread
Times_prob_immune=np.zeros((10,1)) # To record different prob_immune
Times_prob_fatal=np.zeros((10,1)) # To record different prob_fatal
nx = 5
ny = 5
nsteps = 16

# -------------- The function of disease spread --------------
def disease_spread(nx,ny,prob_spread,prob_immune,prob_fatal,nsteps,cny,cnx,crowd,Times_recovered,Times_setvirus,Times_sim,Ratio_immune,Ratio_fatal):

    # # Loop in the "x" direction:
    for k in range(nsteps-1):
        t=0
        for i in range(ny):
    # # Loop in the "y" direction:
            for j in range(nx):
    ## Perform logic here:
                if crowd[k,i,j] == 3:
                    if k+1<nsteps:
                        if np.random.rand()>prob_fatal:
                            for t in range(k+1,nsteps):
                                crowd[t,i,j] = 1
                        else:
                            for t in range(k+1,nsteps):
                                crowd[t,i,j] = 0
                    if 0<=i-1 and crowd[k,i-1,j] == 2:
                        if np.random.rand()<prob_spread and k+1<nsteps:
                            crowd[k+1,i-1,j] = 3
                    if 0<=j-1 and crowd[k,i,j-1] == 2:
                        if np.random.rand()<prob_spread and k+1<nsteps:
                            crowd[k+1,i,j-1] = 3
                    if i+1<ny and crowd[k,i+1,j] == 2:
                        if np.random.rand()<prob_spread and k+1<nsteps:
                            crowd[k+1,i+1,j] = 3
                    if j+1<nx and crowd[k,i,j+1] == 2:
                        if np.random.rand()<prob_spread and k+1<nsteps:
                            crowd[k+1,i,j+1] = 3
        for issick_y in range(ny):
            for issick_x in range(nx):
                if k+1<nsteps and crowd[k+1,issick_y,issick_x] == 3:
                    t=t+1
                    break
        if t == 0:
            Times_recovered[Times_setvirus,Times_sim-1] = k 
            break
    Times_immune=0
    Times_fatal=0
    for i in range(ny):
        for j in range(nx):
            if crowd[k+1,i,j] == 1:
                Times_immune = Times_immune+1
            if crowd[k+1,i,j] == 0:
                Times_fatal = Times_fatal+1
    Ratio_immune[Times_setvirus,Times_sim-1] = Times_immune/(nx*ny)
    Ratio_fatal[Times_setvirus,Times_sim-1] = Times_fatal/(nx*ny)

    # -------------- codes for plot --------------

    # Generate our custom segmented color map for this project.
    # We can specify colors by names and then create a colormap that only uses
    # those names. We have 3 funadmental states, so we want only 3 colors.
    # Color info: https://matplotlib.org/stable/gallery/color/named_colors.html
    
    # Given our "crowd" object, a 2D array that contains 0, 1, 2, or 3,
    # Plot this using the "pcolor" method. Be sure to use our color map and
    # set both *vmin* and *vmax*:
    # ax.pcolor(crowd, cmap=crowd_cmap, vmin=1, vmax=3)
    crowd_cmap = ListedColormap(['black', 'skyblue','white', 'red'])
    # Create figure and set of axes:
    steps=0
    limit_steps=int(np.floor(np.sqrt(nsteps)))
    fig, axs = plt.subplots(limit_steps,limit_steps, figsize=(9, 9))
    for x_fig in range(limit_steps):
        for y_fig in range(limit_steps):       
            contour = axs[x_fig,y_fig].matshow(crowd[steps,:,:],vmin=0,vmax=3,cmap=crowd_cmap)
            axs[x_fig,y_fig].tick_params(axis='x', labelsize=12)
            axs[x_fig,y_fig].tick_params(axis='y', labelsize=12)
            axs[x_fig,y_fig].set_title(f'Steps={steps}',fontsize=12)
            plt.colorbar(contour)
            steps=steps+1

    plt.suptitle('Disease spread 'f'\nprob_spread={prob_spread}, 'f'prob_immue={prob_immune}, 'f'prob_fatal={prob_fatal}\n',fontsize=15)
    plt.tight_layout()
    plt.show()

# -------------- The function of simulation --------------
def crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal):
    
    for Times_setvirus in range(0,Times_confirm):

        # Create an initial grid, set all values to "2". dtype sets the value
        # type in our array to integers only.
        crowd = np.zeros([nsteps, ny, nx], dtype=int) + 2

        # Loop over every cell in the x and y directions.
        for i in range(nx):
            for j in range(ny):
                # Roll our "dice" to see if we get a immune spot:
                if np.random.rand() < prob_immune:
                    for k in range(0,nsteps):
                        crowd[k, i, j] = 1 # 1 is a immune spot.

        # Set the center cell to "sick":
        cny=int((ny-1)/2)
        cnx=int((nx-1)/2)

        # Set disease randomly
        # cny=np.random.randint(0,ny-1)
        # cnx=np.random.randint(0,nx-1)
        crowd[0, cny, cnx] = 3
        disease_spread(nx,ny,prob_spread,prob_immune,prob_fatal,nsteps,cny,cnx,crowd,Times_recovered,Times_setvirus,Times_sim,Ratio_immune,Ratio_fatal)
    Times_prob_spread[Times_sim-1,0]=prob_spread
    Times_prob_immune[Times_sim-1,0]=prob_immune
    Times_prob_fatal[Times_sim-1,0]=prob_fatal

Times_sim = 1
prob_spread = 1
prob_immune = 0.3
prob_fatal = 0.2
crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)

