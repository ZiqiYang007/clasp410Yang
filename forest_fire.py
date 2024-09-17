#!/usr/bin/env python3
'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do this...
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.close('all')

# 1: bare
# 2: Froest
# 3: Burning

Times_confirm = 1 # To have significant results for same condition for burnt
Times_burnt=np.zeros((Times_confirm, 10)) # To record how many steps will it take to end up at each condition
Ratio_bare=np.zeros((Times_confirm, 10)) # To record the Final Ratio of bare/whole field
Times_prob_spread=np.zeros((10,1)) # To record different prob_spread
Times_prob_bare=np.zeros((10,1)) # To record different prob_bare
nx = 3 # x scales of forest
ny = 3 # y scales of forest
nsteps = 9 # steps of fire spreading process

# -------------- The function of fire spread --------------
def forest_fire(nx,ny,prob_spread,prob_bare,nsteps,cny,cnx,forest,Times_burnt,Times_setvirus,Times_sim,Ratio_bare):

    # # Loop in the "x" direction:
    for k in range(nsteps-1):
        t=0
        for i in range(ny):
    # # Loop in the "y" direction:
            for j in range(nx):
    ## Perform logic here:
                if forest[k,i,j] == 3:
                    if k+1<nsteps:
                        for t in range(k+1,nsteps):
                            forest[t,i,j] = 1
                    if 0<=i-1 and forest[k,i-1,j] == 2:
                        if np.random.rand()<prob_spread and k+1<nsteps:
                            forest[k+1,i-1,j] = 3
                    if 0<=j-1 and forest[k,i,j-1] == 2:
                        if np.random.rand()<prob_spread and k+1<nsteps:
                            forest[k+1,i,j-1] = 3
                    if i+1<ny and forest[k,i+1,j] == 2:
                        if np.random.rand()<prob_spread and k+1<nsteps:
                            forest[k+1,i+1,j] = 3
                    if j+1<nx and forest[k,i,j+1] == 2:
                        if np.random.rand()<prob_spread and k+1<nsteps:
                            forest[k+1,i,j+1] = 3
        for isburning_y in range(ny):
            for isburning_x in range(nx):
                if k+1<nsteps and forest[k+1,isburning_y,isburning_x] == 3:
                    t=t+1
                    break
        if t == 0:
            Times_burnt[Times_setvirus,Times_sim-1] = k 
            #print(k) 
            break
    Times_bare=0
    for i in range(ny):
        for j in range(nx):
            if forest[k+1,i,j] == 1:
                Times_bare = Times_bare+1
    Ratio_bare[Times_setvirus,Times_sim-1] = Times_bare/(nx*ny)

    # -------------- codes for plot --------------

    # Generate our custom segmented color map for this project.
    # We can specify colors by names and then create a colormap that only uses
    # those names. We have 3 funadmental states, so we want only 3 colors.
    # Color info: https://matplotlib.org/stable/gallery/color/named_colors.html
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    strange = ListedColormap(['white', 'black', 'red'])
    # Create figure and set of axes:
    steps=0
    limit_steps=int(np.floor(np.sqrt(nsteps)))
    fig, axs = plt.subplots(limit_steps,limit_steps)
    for x_fig in range(limit_steps):
        for y_fig in range(limit_steps):       
            # for x_fig in range(2):
            #     for y_fig in range(2):
            contour = axs[x_fig,y_fig].matshow(forest[steps,:,:],vmin=1,vmax=3,cmap=forest_cmap)
            axs[x_fig,y_fig].tick_params(axis='x', labelsize=10)
            axs[x_fig,y_fig].tick_params(axis='y', labelsize=10)
            axs[x_fig,y_fig].set_title(f'Steps={steps}',fontsize=10)
            plt.colorbar(contour)
            steps=steps+1

                # Given our "forest" object, a 2D array that contains numbers 1, 2, or 3,
                # Plot this using the "pcolor" method. Be sure to use our color map and
                # set both *vmin* and *vmax*:
                # ax.pcolor(forest, cmap=forest_cmap, vmin=1, vmax=3)
    
    plt.suptitle(f'prob_spread={prob_spread}',fontsize=10)
    #  f' Times_setvirus={Times_setvirus}'
    #  f' prob_bare={prob_bare}'
    plt.tight_layout()
    plt.show()

# -------------- The function of simulation --------------
def forest_sim(nx, ny, prob_spread, prob_bare, nsteps, Times_sim, Times_confirm, Times_burnt, Times_prob_spread,Ratio_bare,Times_prob_bare):
    for Times_setvirus in range(0,Times_confirm):
        # Create an initial grid, set all values to "2". dtype sets the value
        # type in our array to integers only.
        forest = np.zeros([nsteps, ny, nx], dtype=int) + 2

        # Loop over every cell in the x and y directions.
        for i in range(nx):
            for j in range(ny):
                # Roll our "dice" to see if we get a bare spot:
                if np.random.rand() < prob_bare:
                    for k in range(nsteps):
                        forest[k, i, j] = 1 # 1 is a bare spot.
        # Set the center cell to "burning":
        cny=int((ny-1)/2)
        cnx=int((nx-1)/2)

        # Set fire randomly
        # cny=np.random.randint(0,ny-1)
        # cnx=np.random.randint(0,nx-1)
        forest[0, cny, cnx] = 3 
        # Even though the center could be bare at the beginning, 
        # because we wouldn't like the simulation to stop initially, 
        # we set the fire on the center after we select initial bare region.

        forest_fire(nx,ny,prob_spread,prob_bare,nsteps,cny,cnx,forest,Times_burnt,Times_setvirus,Times_sim,Ratio_bare)
    Times_prob_spread[Times_sim-1,0]=prob_spread
    Times_prob_bare[Times_sim-1,0]=prob_bare

Times_sim = 1
prob_spread = 1
prob_bare = 0
forest_sim(nx, ny, prob_spread, prob_bare, nsteps, Times_sim, Times_confirm, Times_burnt, Times_prob_spread,Ratio_bare, Times_prob_bare)
Times_sim = Times_sim+1
