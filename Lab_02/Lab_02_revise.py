#!/usr/bin/env python3
'''
As set of tools and routines for solving the N-layer atomsphere energey
balance problem and perform some useful analysis

To reproduce figures in lab writte-up: Do the following

    ipython
    run Lab_02_revise.py
    Fig1()
    Fig2()
    Fig3()
    Fig4()
    Fig5()
    Fig6()
    Fig7()
    Fig8()
    Fig9()
    Fig10()
    Fig11()
    Fig12()
    Fig13()
    Fig14()
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib.image as mpimg

def dNdt_comp(t, N, a=1, b=2, c=1, d=3):

    '''
    This function calculates the Lotka-Volterra competition equations for
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
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

    return dN1dt, dN2dt


def dNdt_pp(t, N, a=1, b=2, c=1, d=3):
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
    a, b, c, d : float, defaults=1, 2, 1, 3
    The value of the Lotka-Volterra coefficients.
    Returns
    -------
    dN1dt, dN2dt : floats
    The time derivatives of `N1` and `N2`.   
    '''
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]

    return dN1dt, dN2dt   

def euler_solve(func, dT, N1_init=0.3, N2_init=0.6,a=1,b=2,c=1,d=3, t_final=100.0):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using Euler method

    Parameters
    ----------
    func : function
    A python function that takes `time`, [`N1`, `N2`] as inputs and
    returns the time derivative of N1 and N2.
    N1_init, N2_init : float
    Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
    Largest timestep allowed in years.
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

def solve_rk8(func, dT, N1_init=0.3, N2_init=0.6,a=1,b=2,c=1,d=3, t_final=100.0):
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
    t_final : float, default=100
    Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
    Lotka-Volterra coefficient values
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


def N1N2vsTime_plot(model_function,dT,N1_init=0.3, N2_init=0.6,\
                    a=1,b=2,c=1,d=3, t_final=100.0,\
                    Title_total = True,N1exist = True,N2exist = True,legendexist = True):
    '''
    This function is going to give the plot of the relationship between two species N1, N2 
    and time for different coexisting model

    Parameters:

    model_function: function
        To tell the type of model for plotting, 
        if Competition model, model_function = dNdt_comp, 
        else if Prey-predator model, model_function = dNdt_pp

    from dT to t_final is same with the introduction in euler_solve

    Title_total: bool, default is True
        This is going to decide if we should add title to the whole graph or not. 
        At default situation, we should include title.
    
    N1exist: bool, default is True
        This is going to decide if we should draw the relationship between N1 and time or not
        At default situation, we should plot N1 vs. time.
    
    N2exist: bool, default is True
        This is going to decide if we should draw the relationship between N2 and time or not
        At default situation, we should plot N2 vs. time.
    
    legendexit: bool, default is True
        This is going to decide if we should include legend in our figure or not.
        At default situation, we should include legend.
    
    Return:
    fig1: figure
        This is the plot which shows the relationship between N1, N2 and time 
        with two different solving method in one figure at default condition
    '''
    # Graph parameters

    fsx = 15
    fsy = 8
    legend_size = 20
    psize = 8
    line_width = 5
    markerpattern = 'o'
    Title_size = 30
    Title_size_axis = 25
    Ticks_size = 20

    # Solving matrix

    Euler_comp = euler_solve(model_function,dT,N1_init, N2_init,a,b,c,d,t_final)
    RK8_comp = solve_rk8(model_function,dT,N1_init, N2_init,a,b,c,d,t_final)

    if model_function == dNdt_comp:
        N1_name = 'N1'
        N2_name = 'N2'
        figure_title = 'Competition Model'
        y_range = np.arange(0,1.3,0.2)
        y_upper = 1.3
        y_lowest = -0.1
    elif model_function == dNdt_pp:
        N1_name = 'N1(prey)'
        N2_name = 'N2(predator)'
        figure_title = 'Prey-predator Model'
        y_range = np.arange(0,3.1,0.5)
        y_upper = 3.1
        y_lowest = -0.1
        
    fig1,ax = plt.subplots(1,1,figsize=(fsx,fsy))
    if Title_total==True:  
        plt.title(f"{figure_title}, N1 & N2 vs. time, dT = {dT}",size = Title_size)
    if N1exist == True:
        ax.plot(Euler_comp[0],Euler_comp[1],label=f'{N1_name}, Euler', color='blue',\
                 linewidth=line_width, marker = markerpattern, markersize = psize)
        ax.plot(RK8_comp[0],RK8_comp[1],label=f'{N1_name}, RK8', linestyle = '--', color='lightblue', \
                linewidth=line_width, marker = markerpattern, markersize = psize)
    if N2exist == True:
        ax.plot(RK8_comp[0],RK8_comp[2],label=f'{N2_name}, RK8', linestyle = '--', color='salmon', \
                linewidth=line_width, marker = markerpattern, markersize = psize)
        ax.plot(Euler_comp[0],Euler_comp[2],label=f'{N2_name}, Euler', color='red', \
                linewidth=line_width, marker = markerpattern, markersize = psize)
    ax.text(0.5, 0.9, f"N1(0)={N1_init}, N2(0)={N2_init}\na={a}, b={b}, c={c}, d={d}", transform=ax.transAxes,\
            fontsize=legend_size, color='black', ha='center',\
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.2))
    plt.xlabel('Time (years)', size = Title_size_axis)
    plt.ylabel('Population', size = Title_size_axis)
    plt.ylim(y_lowest,y_upper)
    plt.yticks(y_range)
    plt.tick_params(axis='both', which='major', labelsize=Ticks_size)

    plt.grid(True)
    if legendexist == True:
        plt.legend(loc = 'upper left',fontsize = legend_size)
    return fig1

def N1N2_phase(model_function,dT,N1_init=0.3, N2_init=0.6,a=1,b=2,c=1,d=3, t_final=100.0):
    '''
    This function is going to give the plot of the relationship between two species N1 and N2 for different coexisting model

    Parameters:

    model_function: function
        To tell the type of model for plotting, 
        if Competition model, model_function = dNdt_comp, 
        else if Prey-predator model, model_function = dNdt_pp

    from dT to t_final is same with the introduction in euler_solve above

    Return:
    fig1: figure
        This is the plot which shows the relationship between N1 and N2 when we use Euler method to solva the equation
    fig2: figure
        This is the plot which shows the relationship between N1 and N2 when we use RK8 method to solva the equation
    '''

    # Graph parameters
    fsx = 12
    fsy = 8
    legend_size = 20
    psize = 8
    line_width = 5
    markerpattern = 'o'
    Title_size = 30
    Title_size_axis = 25
    Ticks_size = 20

    # Solving matrix    

    Euler_comp = euler_solve(model_function,dT,N1_init, N2_init,a,b,c,d,t_final)
    RK8_comp = solve_rk8(model_function,dT,N1_init, N2_init,a,b,c,d,t_final)

    if model_function == dNdt_comp:
        N1_name = 'N1'
        N2_name = 'N2'
        figure_title = 'Competition Model'
    elif model_function == dNdt_pp:
        N1_name = 'N1(prey)'
        N2_name = 'N2(predator)'
        figure_title = 'Prey-predator Model'

    fig1 = plt.figure(figsize = (fsx,fsy))
    ax1 = fig1.add_subplot(111)
    ax1.set_title(f"{figure_title}, N1 vs. N2, dT = {dT}",size = Title_size)
    ax1.plot(Euler_comp[1],Euler_comp[2], color='violet',label = 'Euler', linewidth=line_width, marker = markerpattern, markersize = psize)

    ax1.set_xlabel(f"{N1_name}", size = Title_size_axis)
    ax1.set_ylabel(f"{N2_name}", size = Title_size_axis)
    ax1.tick_params(axis='both', which='major', labelsize=Ticks_size)

    ax1.grid(True)
    ax1.legend(loc = 'upper right',fontsize = legend_size)
    ax1.text(0.5, 0.5, f"N1(0)={N1_init}, N2(0)={N2_init}\na={a}, b={b}, c={c}, d={d}", transform=ax1.transAxes,\
            fontsize=legend_size, color='black', ha='center',\
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.2))

    fig2 = plt.figure(figsize = (fsx,fsy))
    ax2 = fig2.add_subplot(111)
    ax2.set_title(f"{figure_title}, N1 vs. N2, dT = {dT}",size = Title_size)
    ax2.plot(RK8_comp[1],RK8_comp[2], color='orange',label = 'RK8', linewidth=line_width, marker = markerpattern, markersize = psize)
    ax2.set_xlabel(f"{N1_name}", size = Title_size_axis)
    ax2.set_ylabel(f"{N2_name}", size = Title_size_axis)
    ax2.tick_params(axis='both', which='major', labelsize=Ticks_size)

    ax2.grid(True)
    ax2.legend(loc = 'upper right',fontsize = legend_size)
    ax2.text(0.5, 0.5, f"N1(0)={N1_init}, N2(0)={N2_init}\na={a}, b={b}, c={c}, d={d}", transform=ax2.transAxes,\
        fontsize=legend_size, color='black', ha='center',\
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.2))

    return fig1, fig2

def Fig1(func = dNdt_comp,dT=1,code=1):
    '''
    This function is going to generate the Fig 1 in the report and save it as the name "Fig1.png"

    Parameters:
    func: function, default is dNdt_comp
        choose the model function we used for studying the coexistence of two species N1 and N2
    dT: float, default is 1
        The time interval for solving the ODE by numeric method
    code: The number of this figure given by the function, default is 1
    '''
    # Generate Fig1 and save it as "Fig1.png"
    fig = N1N2vsTime_plot(dNdt_comp,dT=1)
    fig.savefig(f"Fig{code}.png",dpi = 500)
    plt.close("all")

def Fig2(func = dNdt_comp, code = 2, dT=[0.1,0.01],N1_0 = [0.3,0.3],N2_0 = [0.6,0.6],\
         a = [1,1],b=[2,2], c=[1,1],d=[3,3],\
            Title_total=True, codeexist = True, subcode=['A','B'],\
                N1E = [True,True], N2E = [True, True],legendexist = True):

    '''
    This function is going to generate the Fig 2 in the report and save it as the name "Fig2.png"
    This function also provide a model for plotting two figures in a row, 
    so there are some parameters for generating two figures respectively.

    Parameters:

    func: function, default is dNdt_comp
        Choose the model function we used for studying the coexistence of two species N1 and N2.
    code: Integer, default is 1
        The number of this figure given by the function.
    dT: float array, default is [0.1,0.01]
        The time intervals for two figures respectively.
    N1_0, N2_0: float array, N1_0's default is [0.3,0.3], N2_0's default is [0.6,0.6]
        Initial conditions for `N1` and `N2`.
    a,b,c,d: float array, defaults are [1,1], [2,2], [1,1], [3,3] respectively
        parameters for the ODE euqation in competition model and prey-predator model.
    Title_total: bool, default is True
        Decide if we give the title of figure or not. True means we give.
    codeexist: bool default is True
        Decide if we give the number of figure or not. True means we give.
    subcode: char array, default is ['A','B']
        The code number of two figures.
    N1E: bool array, defaults is [True, True]
        Decide if we plot the relationship between N1 and time or not. True means we plot.
    N1E: bool array, defaults is [True, True]
        Decide if we plot the relationship between N2 and time or not. True means we plot.
    legendexist: bool, default is True
        Decide if we let the legend show or not. True means we show.

    '''
    # Generate and save subplots A and B respectively
    fig2A = N1N2vsTime_plot(func,dT=dT[0],N1_init=N1_0[0],N2_init=N2_0[0],\
                            a=a[0],b=b[0],c=c[0],d=d[0],Title_total=Title_total,\
                            N1exist=N1E[0],N2exist=N2E[0],legendexist=legendexist)
    fig2B = N1N2vsTime_plot(func,dT=dT[1],N1_init=N1_0[1],N2_init=N2_0[1],\
                            a=a[1],b=b[1],c=c[1],d=d[1],Title_total=Title_total,\
                            N1exist=N1E[1],N2exist=N2E[1],legendexist=legendexist)
    fig2A.savefig(f'fig{code}A.png',dpi = 500)
    fig2B.savefig(f'fig{code}B.png',dpi = 500)

    # The parameters of figure
    fsx = 12
    fsy = 4
    legend_size = 15
    fig,axes = plt.subplots(1,2,figsize = (fsx,fsy))

    # Import subplots
    img1 = mpimg.imread(f'fig{code}A.png')
    axes[0].imshow(img1)
    axes[0].axis('off')

    img2 = mpimg.imread(f'fig{code}B.png')
    axes[1].imshow(img2)
    axes[1].axis('off')

    # Add the number of subplots
    if codeexist == True:
        axes[0].text(0.05, 0.95, subcode[0], transform=axes[0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
        axes[1].text(0.05, 0.95, subcode[1], transform=axes[1].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    # Generate the final figure and save
    plt.tight_layout()
    fig.savefig(f"Fig{code}.png",dpi = 500)
    plt.close("all")

def Fig3():
    '''
    This function is going to generate the Fig 3 in the report and save it as the name "Fig3.png".
    '''
    Fig1(func = dNdt_pp,dT = 0.05,code = 3)

def Fig4():
    '''
    This function is going to generate the Fig 4 in the report and save it as the name "Fig4.png".
    ''' 
    Fig2(func = dNdt_pp,code = 4)

def Fig5(func = dNdt_comp,code = 5):
    '''
    This function is going to generate the Fig 5 in the report and save it as the name "Fig5.png".

    Parameters:
    func: function, default is dNdt_comp
        Choose the model function we used for studying the coexistence of two species N1 and N2.
    code: The number of this figure given by the function, default is 2
    ''' 

    # Set the code number of four subplots

    subcode =np.array([['A','B'],
               ['C','D']])
    
    # Generate and store four subplots
    fig5 = np.empty((2, 2), dtype=object)

    fig5[0,0] = N1N2vsTime_plot(func,dT=1,a=2)
    fig5[0,1] = N1N2vsTime_plot(func,dT=1,b=1)
    fig5[1,0] = N1N2vsTime_plot(func,dT=1,c=2)
    fig5[1,1] = N1N2vsTime_plot(func,dT=1,d=1)

    # Graph parameters
    fsx = 12
    fsy = 8
    legend_size = 15
    fig,axes = plt.subplots(2,2,figsize = (fsx,fsy))

    # Merge four subplots together

    for i in range(2):
        for j in range(2):
            fig5[i,j].savefig(f'fig{code}{subcode[i,j]}.png',dpi = 500)
            img = mpimg.imread(f'fig{code}{subcode[i,j]}.png')
            axes[i,j].imshow(img)
            axes[i,j].axis('off')
            axes[i,j].text(0.05, 0.95, subcode[i,j], transform=axes[i,j].transAxes,\
                           fontsize=legend_size, fontweight='bold', color='black')
    
    # Save the final figure 5
    plt.tight_layout()
    fig.savefig(f"Fig{code}.png",dpi = 500)
    plt.close("all")

def Fig6():
    '''
    This function is going to generate the Fig 6 in the report and save it as the name "Fig6.png".
    '''  
    Fig2(func = dNdt_comp, code = 6, dT = [1,1], c = [0.99999,1], d = [3,3.00001])

def Fig7():
    '''
    This function is going to generate the Fig 7 in the report and save it as the name "Fig7.png".
    '''    
    Fig2(func = dNdt_comp, code = 7, dT = [1,1], N1_0 = [0.3,0.300001], N2_0 = [0.600001,0.6])

def Fig8():
    '''
    This function is going to generate the Fig 8 in the report and save it as the name "Fig8.png".
    '''      
    Fig2(func = dNdt_comp, code = 8, dT=[1,1],N1_0 = [0.5,0.5],N2_0 = [0.5,0.5],\
         a = [1,1],b=[2,2], c=[1,1],d=[2,2],\
            Title_total=True, codeexist = True, subcode=['A','B'],N1E = [True,False], N2E = [False, True])
    
def Fig9(code1='9up',code2='9down',code=9):
    '''
    This function is going to generate the Fig 9 in the report and save it as the name "Fig9.png".

    Parameters:
    code1: char, default is '9up'
        the name of upper two figures
    code2: char, default is '9down'
        the name of low two figures
    ''' 
    # Generate upper and low figures  
    Fig2(func = dNdt_comp, code = code1, dT=[1,1],N1_0 = [0.5,0.5],N2_0 = [0.5,0.5],\
         a = [1,1],b=[2,2], c=[1,1],d=[2,2],\
            Title_total=True, codeexist = True, subcode=['A','B'],N1E = [True,False], N2E = [False, True])
    Fig2(func = dNdt_comp, code = code2, dT=[1,1],N1_0 = [0.2,1/7],N2_0 = [0.4,3/7],\
         a = [1,1],b=[2,2], c=[1,0.5],d=[3,2],\
            Title_total=True, codeexist = True, subcode=['C','D'],N1E = [True,True], N2E = [True, True],legendexist=False)
    
    fig,axes = plt.subplots(2,1)

    # import two figures and combine them together
    img1 = mpimg.imread(f'Fig{code1}.png')
    axes[0].imshow(img1)
    axes[0].axis('off')

    img2 = mpimg.imread(f'Fig{code2}.png')
    axes[1].imshow(img2)
    axes[1].axis('off')

    # Generate the final figure and save it as "Fig10.png"
    plt.tight_layout()
    fig.savefig(f"Fig{code}.png",dpi = 500)
    plt.close("all")

def Fig10():
    '''
    This function is going to generate the Fig 10 in the report and save it as the name "Fig10.png".
    '''    
    Fig2(func = dNdt_pp,code = 10, dT = [0.05,0.05],d=[2,4])

def Fig11():
    '''
    This function is going to generate the Fig 11 in the report and save it as the name "Fig11.png".
    '''    
    Fig2(func = dNdt_pp,code = 11, dT = [0.05,0.05],b=[1,3])

def Fig12():
    '''
    This function is going to generate the Fig 12 in the report and save it as the name "Fig12.png".
    ''' 

    Fig2(func = dNdt_pp,code = 12, dT = [0.05,0.05],N1_0=[0.6,0.6],N2_0=[0.3,0.3],b=[1,3])

def Fig13(code = 13,dT = [0.01,0.05],d = [3,3]):
    '''
    This function is going to generate the Fig 13 in the report and save it as the name "Fig13.png".

    Parameters:
    code: integer, default is 13
        The number of this figure given by the function.
    dT: float array, default is [0.01,0.05]
        The time intervals for two figures at the same row respectively.
    d: float array, default is [3,3]
        parameter d for the ODE euqation in competition model and prey-predator model.
    '''

    # Generate subplots and save them
    legend_size = 8
    fig,axes = plt.subplots(2,2)
    fig13A,fig13C = N1N2_phase(model_function = dNdt_pp,dT = dT[0],d = d[0])
    fig13B,fig13D = N1N2_phase(model_function = dNdt_pp,dT = dT[1],d = d[1])

    fig13A.savefig(f"fig{code}A.png",dpi = 500)
    fig13B.savefig(f"fig{code}B.png",dpi = 500)
    fig13C.savefig(f"fig{code}C.png",dpi = 500)
    fig13D.savefig(f"fig{code}D.png",dpi = 500)

    # Set the code of subplots respectively
    subcode = np.array([['A','B'],
                ['C','D']])
    
    # Combine four subplots together
    for i in [0,1]:
        for j in [0,1]:
            img = mpimg.imread(f'fig{code}{subcode[i,j]}.png')
            axes[i,j].imshow(img)
            axes[i,j].axis('off')
            axes[i,j].text(0.05, 0.95, subcode[i,j], transform=axes[i,j].transAxes,\
                 fontsize=legend_size, fontweight='bold', color='black')
    
    # Save the final figure 13 as "Fig13.png"
    plt.tight_layout()
    fig.savefig(f"Fig{code}.png",dpi = 500)
    plt.close("all")

def Fig14():
    '''
    This function is going to generate the Fig 14 in the report and save it as the name "Fig14.png".
    '''  
    Fig13(code = 14, dT = [0.05,0.05], d = [2,4])