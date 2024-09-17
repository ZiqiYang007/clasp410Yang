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

Times_confirm = 50 # To have significant results for same condition for recovered
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


# -------------- Fatality rates vs.  Recovered ratio --------------
# prob_spread = 1, prob_immune = 0 Ratio vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 1
prob_immune = 0
for prob_fatal in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal*0.1, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

fig1 = plt.figure()
ax = fig1.add_subplot(111) # The (111) here will be discussed in later labs; just include it for now.
plt.title("Final Ratio of immune  vs. fatality rates",size = 15)
ax.plot(Times_prob_fatal, Ratio_immune_average, label=f'prob_spread = {prob_spread}, prob_immune = {prob_immune}', color='blue', linewidth=2, marker = 'o')

# prob_spread = 0.5, prob_immune = 0 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 0.5
prob_immune = 0
for prob_fatal in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal*0.1, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_fatal, Ratio_immune_average, label=f'prob_spread = {prob_spread}, prob_immune = {prob_immune}', color='orange', linewidth=2, marker = 'o')

# prob_spread = 1, prob_immune = 0.5 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 1
prob_immune = 0.5
for prob_fatal in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal*0.1, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_fatal, Ratio_immune_average, label=f'prob_spread = {prob_spread}, prob_immune = {prob_immune}', color='green', linewidth=2, marker = 'o')

# prob_spread = 0.5, prob_immune = 0.5 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 0.5
prob_immune = 0.5
for prob_fatal in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal*0.1, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_fatal, Ratio_immune_average, label=f'prob_spread = {prob_spread}, prob_immune = {prob_immune}', color='pink', linewidth=2, marker = 'o')

plt.xlabel('Fatality rates', size = 15)
plt.ylabel('Final Ratio of immune', size = 15)

ax.legend()
plt.show()


# # -------------- Fatality rates vs.  Iteration recovered --------------
# independent variation: initial immune probability
# dependent variation: how many steps will take as the disease disappear (Times_recovered_average)
# dependent variation: The ratio of recovered field/whole field at the end (Ratio_immune_average). 

# prob_spread = 1, prob_immune = 0 Ratio vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 1
prob_immune = 0
for prob_fatal in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal*0.1, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

fig1 = plt.figure()
ax = fig1.add_subplot(111) # The (111) here will be discussed in later labs; just include it for now.
plt.title("Step as disease stops spreading  vs. fatality rates",size = 15)
ax.plot(Times_prob_fatal, Times_recovered_average, label=f'prob_spread = {prob_spread}, prob_immune = {prob_immune}', color='blue', linewidth=2, marker = 'o')

# prob_spread = 0.5, prob_fatal = 0 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 0.5
prob_immune = 0
for prob_fatal in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal*0.1, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_fatal, Times_recovered_average, label=f'prob_spread = {prob_spread}, prob_immune = {prob_immune}', color='orange', linewidth=2, marker = 'o')

# prob_spread = 1, prob_fatal = 0.5 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 1
prob_immune = 0.5
for prob_fatal in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal*0.1, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_fatal, Times_recovered_average, label=f'prob_spread = {prob_spread}, prob_immune = {prob_immune}', color='green', linewidth=2, marker = 'o')

# prob_spread = 0.5, prob_fatal = 0.5 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 0.5
prob_immune = 0.5
for prob_fatal in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune, prob_fatal*0.1, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_fatal, Times_recovered_average, label=f'prob_spread = {prob_spread}, prob_immune = {prob_immune}', color='pink', linewidth=2, marker = 'o')

plt.xlabel('Fatality rates', size = 15)
plt.ylabel('Step as disease stops spreading', size = 15)

ax.legend()
plt.show()

# # -------------- Initial immune probability vs.  Recovered ratio --------------
# independent variation: initial immune probability
# dependent variation: how many steps will take as the disease disappear (Times_recovered_average)
# dependent variation: The ratio of recovered field/whole field at the end (Ratio_immune_average). 

# prob_spread = 1, prob_fatal = 0 Ratio vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 1
prob_fatal = 0
for prob_immune in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune*0.1, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

fig1 = plt.figure()
ax = fig1.add_subplot(111) # The (111) here will be discussed in later labs; just include it for now.
plt.title("Final Ratio of immune  vs. early vaccine rates",size = 15)
ax.plot(Times_prob_immune, Ratio_immune_average, label=f'prob_spread = {prob_spread}, prob_fatal = {prob_fatal}', color='blue', linewidth=2, marker = 'o')

# prob_spread = 0.5, prob_fatal = 0 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 0.5
prob_fatal = 0
for prob_immune in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune*0.1, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_immune, Ratio_immune_average, label=f'prob_spread = {prob_spread}, prob_fatal = {prob_fatal}', color='orange', linewidth=2, marker = 'o')

# prob_spread = 1, prob_fatal = 0.5 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 1
prob_fatal = 0.5
for prob_immune in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune*0.1, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_immune, Ratio_immune_average, label=f'prob_spread = {prob_spread}, prob_fatal = {prob_fatal}', color='green', linewidth=2, marker = 'o')

# prob_spread = 0.5, prob_fatal = 0.5 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 0.5
prob_fatal = 0.5
for prob_immune in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune*0.1, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_immune, Ratio_immune_average, label=f'prob_spread = {prob_spread}, prob_fatal = {prob_fatal}', color='pink', linewidth=2, marker = 'o')

plt.xlabel('Early vaccine rates', size = 15)
plt.ylabel('Final Ratio of immune', size = 15)

ax.legend()
plt.show()

# # -------------- Initial immune probability vs.  Iteration recovered --------------
# independent variation: initial immune probability
# dependent variation: how many steps will take as the disease disappear (Times_recovered_average)
# dependent variation: The ratio of recovered field/whole field at the end (Ratio_immune_average). 

# prob_spread = 1, prob_fatal = 0 Ratio vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 1
prob_fatal = 0
for prob_immune in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune*0.1, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

fig1 = plt.figure()
ax = fig1.add_subplot(111) # The (111) here will be discussed in later labs; just include it for now.
plt.title("Step as disease stops spreading  vs. early vaccine rates",size = 15)
ax.plot(Times_prob_immune, Times_recovered_average, label=f'prob_spread = {prob_spread}, prob_fatal = {prob_fatal}', color='blue', linewidth=2, marker = 'o')

# prob_spread = 0.5, prob_fatal = 0 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 0.5
prob_fatal = 0
for prob_immune in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune*0.1, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_immune, Times_recovered_average, label=f'prob_spread = {prob_spread}, prob_fatal = {prob_fatal}', color='orange', linewidth=2, marker = 'o')

# prob_spread = 1, prob_fatal = 0.5 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 1
prob_fatal = 0.5
for prob_immune in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune*0.1, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_immune, Times_recovered_average, label=f'prob_spread = {prob_spread}, prob_fatal = {prob_fatal}', color='green', linewidth=2, marker = 'o')

# prob_spread = 0.5, prob_fatal = 0.5 Steps vs. immune probability
Times_sim = 1
Times_recovered_average = np.zeros((1,10))
prob_spread = 0.5
prob_fatal = 0.5
for prob_immune in range (1,11):
    crowd_sim(nx, ny, prob_spread, prob_immune*0.1, prob_fatal, nsteps, Times_sim, Times_confirm, Times_recovered, Times_prob_spread,Ratio_immune,Ratio_fatal,Times_prob_immune,Times_prob_fatal)
    Times_sim = Times_sim+1
Times_recovered_average = np.mean(Times_recovered,axis=0)
Ratio_immune_average = np.mean(Ratio_immune,axis=0)

ax.plot(Times_prob_immune, Times_recovered_average, label=f'prob_spread = {prob_spread}, prob_fatal = {prob_fatal}', color='pink', linewidth=2, marker = 'o')

plt.xlabel('Early vaccine rates', size = 15)
plt.ylabel('Step as disease stops spreading', size = 15)

ax.legend()
plt.show()
# # introduction, methodology, result, discussion