import numpy as np
import matplotlib.pyplot as plt

def dNdt_comp(t, N, a, b, c, d):

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

    return dN1dt, dN2dt  

def euler_solve(func, dT, N1_init, N2_init, a, b, c, d, t_final=100.0):

    time = np.arange(0,t_final+dT,dT)
    N1 = np.zeros(time.size)
    N2 = np.zeros(time.size)
    N1[0] = N1_init
    N2[0] = N2_init 

    # Important code goes here #
    for i in range(1, time.size):
        dN1, dN2 = func(i, [N1[i-1], N2[i-1]],a,b,c,d)
        N1[i] = N1[i-1]+dT*dN1
        N2[i] = N2[i-1]+dT*dN2
    
    return time, N1, N2


def solve_rk8(func, dT, N1_init, N2_init, a, b, c, d, t_final=100.0):

    from scipy.integrate import solve_ivp
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
    args=[a, b, c, d], method='DOP853', max_step=dT)

    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    # Return values to caller.
    return time, N1, N2

# Graph parameters
psize = 3
line_width = 2
markerpattern = 'o'
N1_initial = 0.5
N2_initial = 0.5

a = 1
b = 2
c = 1
d = 2
dT = 1

Euler_comp = euler_solve(dNdt_comp,dT,N1_initial,N2_initial,a,b,c,d)
RK8_comp = solve_rk8(dNdt_comp,dT,N1_initial,N2_initial,a,b,c,d)

fig1 = plt.figure()
ax = fig1.add_subplot(111)  
plt.title(f"Competition Model, N1, N2 vs. time \n N1_initial = {N1_initial}; N2_initial = {N2_initial} \n a = {a};  b = {b}; c = {c}; d = {d}; dT = {dT}"
          ,size = 15)
ax.plot(Euler_comp[0],Euler_comp[1],label='N1, Euler', color='blue', linewidth=line_width, marker = markerpattern, markersize = psize)
#ax.plot(Euler_comp[0],Euler_comp[2],label='N2, Euler', color='red', linewidth=line_width, marker = markerpattern, markersize = psize)
ax.plot(RK8_comp[0],RK8_comp[1],label='N1, RK8', linestyle = '--', color='lightblue', linewidth=line_width, marker = markerpattern, markersize = psize)
#ax.plot(RK8_comp[0],RK8_comp[2],label='N2, RK8', linestyle = '--', color='salmon', linewidth=line_width, marker = markerpattern, markersize = psize)

plt.xlabel('Time (years)', size = 15)
plt.ylabel('Population', size = 15)

plt.grid(True)
ax.legend()
plt.show()