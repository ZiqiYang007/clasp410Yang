import numpy as np
import matplotlib.pyplot as plt

def dNdt_pp(t, N, a, b, c, d):
    
    '''
    This function calculates the Lotka-Voterra Predator-Prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.
    
    Parameters
    ----------
    t : float
    The current time (not used here).
    N : two-element list
    The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d: float
    The value of the Lotka-Volterra coefficients.
    
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.   
    '''
    
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]

    return dN1dt, dN2dt  

def euler_solve(func, dT, N1_init, N2_init, a, b, c, d, t_final=100.0):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using Euler method

    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    dT : float
    Timestep in years.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    a, b, c, d: float
    The value of the Lotka-Volterra coefficients.
    t_final : float, default=100
    Integrate until this value is reached, in years.

    Returns
    -------
    time : Numpy array
    Time elapsed in years.
    N1, N2 : Numpy arrays
    Normalized population density solutions.
    '''
    
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
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    
    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
    Largest timestep allowed in years.
    a, b, c, d : float
    Lotka-Volterra coefficient values
    t_final : float, default=100
    Integrate until this value is reached, in years.
    
    Returns
    -------
    time : Numpy array
    Time elapsed in years.
    N1, N2 : Numpy arrays
    Normalized population density solutions.
    '''
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
N1_initial = 0.3
N2_initial = 0.6

a = 1
b = 2
c = 1
d = 3
dT = 0.01

Euler_pp = euler_solve(dNdt_pp,dT,N1_initial,N2_initial,a,b,c,d)
RK8_pp = solve_rk8(dNdt_pp,dT,N1_initial,N2_initial,a,b,c,d)

fig1 = plt.figure()
ax = fig1.add_subplot(111)  
plt.title(f"Prey-predator Model, N1, N2 vs. time \n N1_initial = {N1_initial}; N2_initial = {N2_initial} \n a = {a};  b = {b}; c = {c}; d = {d}; dT = {dT}"
          ,size = 15)
ax.plot(Euler_pp[0],Euler_pp[1],label='N1 (prey), Euler', color='blue', linewidth=line_width, marker = markerpattern, markersize = psize)
ax.plot(Euler_pp[0],Euler_pp[2],label='N2 (predator), Euler', color='red', linewidth=line_width, marker = markerpattern, markersize = psize)
ax.plot(RK8_pp[0],RK8_pp[1],label='N1 (prey), RK8', linestyle = '--', color='lightblue', linewidth=line_width, marker = markerpattern, markersize = psize)
ax.plot(RK8_pp[0],RK8_pp[2],label='N2 (predator), RK8', linestyle = '--', color='salmon', linewidth=line_width, marker = markerpattern, markersize = psize)

plt.xlim([0, 100])
plt.ylim([0, 3])
plt.xlabel('Time (years)', size = 15)
plt.ylabel('Population', size = 15)
plt.grid(True)
ax.legend()

fig2 = plt.figure()
ax = fig2.add_subplot(111)
plt.title(f"Prey-predator Model predator vs. prey \n N1_initial = {N1_initial}; N2_initial = {N2_initial} \n a = {a};  b = {b}; c = {c}; d = {d}; dT = {dT}"
          ,size = 15)
ax.plot(Euler_pp[1],Euler_pp[2],label='Euler', color='blue', linewidth=line_width, marker = markerpattern, markersize = psize)
plt.grid(True)
ax.legend()

fig3 = plt.figure()
ax = fig3.add_subplot(111)
plt.title(f"Prey-predator Model predator vs. prey \n N1_initial = {N1_initial}; N2_initial = {N2_initial} \n a = {a};  b = {b}; c = {c}; d = {d}; dT = {dT}"
          ,size = 15)
ax.plot(RK8_pp[1],RK8_pp[2],label='RK8', color='red', linewidth=line_width, marker = markerpattern, markersize = psize)

plt.grid(True)
ax.legend()
plt.show()
