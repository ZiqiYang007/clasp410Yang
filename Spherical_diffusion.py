'''
This code is for the volatiles diffusion in spherical glass bead

To produce all figures in the report, please do these:

    ipython
    run Spherical_diffusion.py
    Diffusionmap_NACS() # Generate Fig2map and Fig2FP
    Diffusionmap_NACS_Fig3() # Generate Fig3map and Fig3FP
    Na_diff_vary_boundary() # Generate Fig4map and Fig4FP
    Na_diff_increase_boundary() # Generate Fig5map and Fig5FP
    Na_diff_increase_boundary_speed_up() # Generate Fig6map, Fig6FP and Fig7FP
    Na_diff_vary_boundary_temperature_decrease() # Generate Fig8map and Fig8FP
    Na_diff_increase_boundary_temperature_decrease() # Generate Fig10map and Fig10FP (We will only use Fig10FP and it will be called Fig. 10 in the report)
    Na_diff_increase_boundary_exp_temperature_decrease() # Generate Fig11map, Fig11FP (We will only use Fig11FP and it will be called Fig. 11 in the report)

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.special import erfc
import matplotlib.image as mpimg

MAl2O3 = 2*26.98+3*16
MCaO = 40.08+16
MNa2O = 22.99*2+16
MS = 32

P_NCAS = np.array([[0.83,-0.32,-0.01],
            [-0.56,0.95,-0.65],
            [0.06,0.04,0.76]]) # The eigenvector of NCAS system, from top to bottom: Al2O3, CaO, Na2O

a_NCAS = [-9.967,-6.195,-13.697]
b_NCAS = [-29267,-32624,-15541]

def Lamada_matrix(T = 1640, lam_dim = 3,a = a_NCAS, b = b_NCAS):
    '''
    This function is to generate the eigenvalues matrix of NCAS system

    ----------
    Parameters
    ----------

    T: float
        temperature of system
    lam_dim: int
        the dimension of the eigenvalues matrix
    a: float array
        the a coefficient of each eigenvalue
    b: float array
        the b coefficient of each eigenvalue
    
    ----------
    Return
    ----------

    eigenvalues: float matrix
        the eigenvalues matrix
    
    '''
    eigenvalues = np.zeros((lam_dim,lam_dim))
    for i in range(lam_dim):
        eigenvalues[i,i] = (10**12)*(np.exp(a[i]+b[i]/T)) # Diffusivity unit (um^2)/s

    return eigenvalues

def analytical_solution(x, t, xmax, D, u0, n_terms):
    '''
    This function is to give the analytical solution of multi-component diffusion

    ----------
    Parameters
    ----------

    x: float
        the position of the point
    t: float
        the time of the point
    xmax: float
        the maximum distance of the system
    D: float
        diffusion coefficient
    u0: float
        initial value
    n_terms: int
        The number of terms to which the infinite series is expanded
    
    ----------
    Return
    ----------

    u: float
        the value of u at certain x and t
    
    '''

    x = np.asarray(x, dtype=float)
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1, 2): 
        coefficient = (2 * u0) / (n * np.pi)
        spatial = np.sin(n * np.pi * x / xmax)
        temporal = np.exp(-D * (n * np.pi / xmax)**2 * t)
        u += coefficient * spatial * temporal
    return u

def Material_diffusion_NACS(T_K = 1773,xmax = 1000,tmax = 100,dx = 10, dt = 0.02,\
                            C0_Al = 11.4, C0_Ca = 10.8,C0_Na = 13.3,\
                            basalts = False):
    '''
    This function is to give the numerical and analytical solution of multi-component diffusion

    ----------
    Parameters
    ----------

    T_K: float, defaults is 1773
        temperature, unit is K
    xmax: float, defaults is 1000
        the diameter of the glass bead, unit is um
    tmax: float, defaults is 100
        the time when diffusion end, unit is seconds
    dx: float, defaults is 10
        the intervals between two nearby x grid, unit is um
    dt: float, defaults is 0.02
        the intervals between two nearby t grid, unit is seconds
    C0_Al, C0_Ca, C0_Na: float, defaults are 11.4, 10.8, 13.3
        initial concentration, unit is %
    basalts: bool, defaults is False
        if this is true, we will include basalts diffusion profile in our analytical
    
    ----------
    Return
    ----------
    xgrid: float array
        a series of points on the diameter of the glass bead, unit is um
    tgrid: float array
        a series of time points, unit is seconds
    CAl, CCa, CNa: float arraies
        concentration arraies of three element in x-t space solved by numerical method, units are %
    CNa_basalts: float array
        concentration array of Na in x-t space in basaltic system, units are %    
    CAl_BT,CCa_BT,CNa_BT: float arraies
        concentration arraies of three element in x-t space solved by analytical method, units are %
    tmax: float
        the time when diffusion end, unit is seconds
    
    '''

    
    P_NCAS_N1 = np.linalg.inv(P_NCAS)
    xgrid, tgrid = np.arange(0,xmax+dx,dx),np.arange(0,tmax+dt,dt)
    M = int(np.round(xmax/dx+1))
    N = int(np.round(tmax/dt+1))

    CAl = np.zeros((M,N))
    CCa = np.zeros((M,N))
    CNa = np.zeros((M,N))

    CAl_BT = np.zeros((M,N))
    CCa_BT = np.zeros((M,N))
    CNa_BT = np.zeros((M,N))  

    CNa_basalts = np.zeros((M,N))

    # Set initial condition

    for i in range(M):
        CAl[i,0] = C0_Al
        CCa[i,0] = C0_Ca
        CNa[i,0] = C0_Na
        if basalts:
            CNa_basalts[i,0] = C0_Na

    # Set boundary condition

    CAl[M-1,:] = 0
    CCa[M-1,:] = 0
    CNa[M-1,:] = 0
    if basalts:
        CNa_basalts[M-1,:] = 0

    CAl[0,:] = 0
    CCa[0,:] = 0
    CNa[0,:] = 0
    if basalts:
        CNa_basalts[0,:] = 0

    CAl_BT = CAl.copy()
    CCa_BT = CCa.copy()
    CNa_BT = CNa.copy()

    u1 = np.zeros((M,N))
    u2 = np.zeros((M,N))
    u3 = np.zeros((M,N))

    # Transfer and set initial condition

    u1[:,0] = P_NCAS_N1[0,0]*CAl[:,0]+P_NCAS_N1[0,1]*CCa[:,0]+P_NCAS_N1[0,2]*CNa[:,0]
    u2[:,0] = P_NCAS_N1[1,0]*CAl[:,0]+P_NCAS_N1[1,1]*CCa[:,0]+P_NCAS_N1[1,2]*CNa[:,0]
    u3[:,0] = P_NCAS_N1[2,0]*CAl[:,0]+P_NCAS_N1[2,1]*CCa[:,0]+P_NCAS_N1[2,2]*CNa[:,0]

    # Transfer and set boundary condition

    u1[0,:] = P_NCAS_N1[0,0]*CAl[0,:]+P_NCAS_N1[0,1]*CCa[0,:]+P_NCAS_N1[0,2]*CNa[0,:]
    u2[0,:] = P_NCAS_N1[1,0]*CAl[0,:]+P_NCAS_N1[1,1]*CCa[0,:]+P_NCAS_N1[1,2]*CNa[0,:]
    u3[0,:] = P_NCAS_N1[2,0]*CAl[0,:]+P_NCAS_N1[2,1]*CCa[0,:]+P_NCAS_N1[2,2]*CNa[0,:]
    u1[M-1,:] = P_NCAS_N1[0,0]*CAl[M-1,:]+P_NCAS_N1[0,1]*CCa[M-1,:]+P_NCAS_N1[0,2]*CNa[M-1,:]
    u2[M-1,:] = P_NCAS_N1[1,0]*CAl[M-1,:]+P_NCAS_N1[1,1]*CCa[M-1,:]+P_NCAS_N1[1,2]*CNa[M-1,:]
    u3[M-1,:] = P_NCAS_N1[2,0]*CAl[M-1,:]+P_NCAS_N1[2,1]*CCa[M-1,:]+P_NCAS_N1[2,2]*CNa[M-1,:]

    # The initial condition of Boltzman transformation solution is same with numerical solution

    u1_BT = u1.copy()
    u2_BT = u2.copy()
    u3_BT = u3.copy()

    # Set eigenvalues (lamada) matrix:

    lamada_matrix = Lamada_matrix(T_K)

    if basalts:
        DNa_basalts = (10**12)*np.exp(-9.251-(19628/T_K))

    # Solve the heat_diffusion
    for j in range(0,N-1):
        for i in range(1,M-1):
            # Numerical solution:
            u1[i,j+1] = dt*lamada_matrix[0,0]*(u1[i+1,j]-2*u1[i,j]+u1[i-1,j])/(dx**2)+u1[i,j]
            u2[i,j+1] = dt*lamada_matrix[1,1]*(u2[i+1,j]-2*u2[i,j]+u2[i-1,j])/(dx**2)+u2[i,j]
            u3[i,j+1] = dt*lamada_matrix[2,2]*(u3[i+1,j]-2*u3[i,j]+u3[i-1,j])/(dx**2)+u3[i,j]

            # Boltzman transformation:
            n_terms = 15
            u1_BT[i,j+1] = analytical_solution(x=i*dx, t=(j+1)*dt, xmax=xmax, D=lamada_matrix[0,0], u0=u1_BT[1,0], n_terms=n_terms)
            u2_BT[i,j+1] = analytical_solution(x=i*dx, t=(j+1)*dt, xmax=xmax, D=lamada_matrix[1,1], u0=u2_BT[1,0], n_terms=n_terms)
            u3_BT[i,j+1] = analytical_solution(x=i*dx, t=(j+1)*dt, xmax=xmax, D=lamada_matrix[2,2], u0=u3_BT[1,0], n_terms=n_terms)

            if basalts:
                CNa_basalts[i,j+1] = dt*DNa_basalts*(CNa_basalts[i+1,j]-2*CNa_basalts[i,j]+CNa_basalts[i-1,j])/(dx**2)+CNa_basalts[i,j]
    for j in range(0,N):
        for i in range(0,M):
            CAl[i,j] = P_NCAS[0,0]*u1[i,j]+P_NCAS[0,1]*u2[i,j]+P_NCAS[0,2]*u3[i,j]
            CCa[i,j] = P_NCAS[1,0]*u1[i,j]+P_NCAS[1,1]*u2[i,j]+P_NCAS[1,2]*u3[i,j]
            CNa[i,j] = P_NCAS[2,0]*u1[i,j]+P_NCAS[2,1]*u2[i,j]+P_NCAS[2,2]*u3[i,j]
            CAl_BT[i,j] = P_NCAS[0,0]*u1_BT[i,j]+P_NCAS[0,1]*u2_BT[i,j]+P_NCAS[0,2]*u3_BT[i,j]
            CCa_BT[i,j] = P_NCAS[1,0]*u1_BT[i,j]+P_NCAS[1,1]*u2_BT[i,j]+P_NCAS[1,2]*u3_BT[i,j]
            CNa_BT[i,j] = P_NCAS[2,0]*u1_BT[i,j]+P_NCAS[2,1]*u2_BT[i,j]+P_NCAS[2,2]*u3_BT[i,j]

    if basalts:
        return xgrid, tgrid, CAl, CCa, CNa, CNa_basalts, CAl_BT,CCa_BT,CNa_BT, tmax
    else:
        return xgrid, tgrid, CAl, CCa, CNa, CAl_BT, CCa_BT, CNa_BT, tmax

def Diffusionmap_NACS(T_K = 1640, xmax = 500,tmax = 500,dx = 10,dt = 0.02,\
                      C0_Al=11.4, C0_Ca = 10.8, C0_Na = 13.3, moll = False,figcode = 2, basalts = False):

    '''
    This function is to generate heatmap picture of Al, Ca and Na element concentration 
    and the final profiles of element concentration.

    ----------
    Parameters
    ----------

    T_K: float, defaults is 1640
        temperature, unit is K
    xmax: float, defaults is 500
        the diameter of the glass bead, unit is um
    tmax: float, defaults is 500
        the time when diffusion end, unit is seconds
    dx: float, defaults is 10
        the intervals between two nearby x grid, unit is um
    dt: float, defaults is 0.02
        the intervals between two nearby t grid, unit is seconds
    C0_Al, C0_Ca, C0_Na: float, defaults are 11.4, 10.8, 13.3
        initial concentration, unit is %
    moll: bool, defaults is False
        If this is true, we will transfer the concentration unit into mol/um^3
    figcode: int, defaults is 2
        The number of figure
    basalts: bool, defaults is False
        if this is true, we will include basalts diffusion profile in our analytical
    
    '''

    # Figure parameters:
    fsx = 8 
    fsy = 6
    legend_size = 18
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 18
    Title_size_axis = 18
    Ticks_size = 18

    V = (xmax/2)**3*4/3*3.14

    if moll:
        C0_Al = C0_Al/(MAl2O3*V) # mol/(um3)
        C0_Ca = C0_Ca/(MCaO*V) # mol/(um3)
        C0_Na = C0_Na/(MNa2O*V) # mol/(um3)

    if basalts:
        x,time,CAl,CCa,CNa,CNa_basalts, CAl_BT,CCa_BT,CNa_BT, tmax = Material_diffusion_NACS(
                                                        T_K = T_K,xmax = xmax,tmax = tmax,dx = dx, dt = dt,\
                                                       C0_Al = C0_Al, C0_Ca = C0_Ca,C0_Na = C0_Na,basalts = basalts)
    else:   
        x,time,CAl,CCa,CNa, CAl_BT,CCa_BT,CNa_BT, tmax = Material_diffusion_NACS(
                                                        T_K = T_K,xmax = xmax,tmax = tmax,dx = dx, dt = dt,\
                                                       C0_Al = C0_Al, C0_Ca = C0_Ca,C0_Na = C0_Na, basalts = basalts)

    if basalts:
        C = [CAl,CCa,CNa,CNa_basalts]
        C_BT = [CAl_BT,CCa_BT,CNa_BT,CNa_basalts]
        C0 = [C0_Al,C0_Ca,C0_Na,C0_Na]
        Title = ['Al','Ca','Na','Na']
        system = ['Diorite','Diorite','Diorite','Basalts']
        color_seq = ['blue','red','green','violet']
        color_seq_BT = ['navy','brown','greenyellow','darkviolet']
    
    else:
        C = [CAl,CCa,CNa]
        C_BT = [CAl_BT,CCa_BT,CNa_BT]
        C0 = [C0_Al,C0_Ca,C0_Na]
        Title = ['Al','Ca','Na']
        system = ['Diorite','Diorite','Diorite']
        color_seq = ['blue','red','green']
        color_seq_BT = ['navy','brown','greenyellow']

    # Diffusion heat map:

    for i in range(len(C)):

        # Create a figure/axes object
        fig,ax= plt.subplots(1,2,figsize=(fsx*2,fsy), constrained_layout=True)
        code = ["A","B"]

        # Create a color map and add a color bar.
        map = ax[0].pcolor(time, x, C[i], cmap='magma', vmin=0, vmax=C0[i], shading='nearest')
        map = ax[1].pcolor(time, x, C_BT[i], cmap='magma', vmin=0, vmax=C0[i], shading='nearest')

        ax[0].set_title(f"{Title[i]} ({system[i]}, numerical)",fontsize = Title_size)
        ax[1].set_title(f"{Title[i]} ({system[i]}, analytical)",fontsize = Title_size)


        for j in range(2):
            ax[j].set_xlabel("Time (s)", fontsize = Title_size_axis)
            ax[j].set_ylabel("Distance (um)", fontsize = Title_size_axis)
            ax[j].tick_params(axis='both', which='major', labelsize = Ticks_size)
            ax[j].set_xlim(0,tmax+dt)
            ax[j].set_xticks(np.arange(0,tmax+dt,tmax/10))
            ax[j].text(0.5, 0.75, f'Initial concentration\nAl:{C0_Al}%\nCa:{C0_Ca}%\nNa:{C0_Na}%',\
                    transform=ax[j].transAxes, fontsize=legend_size, color='black',\
                    bbox = dict(facecolor = 'white',edgecolor = 'black',alpha = 0.6))
            ax[j].text(0.05, 0.95, code[j],\
                    transform=ax[j].transAxes, fontsize=legend_size, color='black',\
                    bbox = dict(facecolor = 'white',edgecolor = 'black',alpha = 0.6))
        
        if basalts and i == 3:
            fig.delaxes(ax[1])

        label = 'Concentration (wt%)'

        if moll:
            label = 'Concentration(mol/um^3)'

        cb = plt.colorbar(map, ax=ax, label=label)
        cb.ax.tick_params(labelsize = Ticks_size)
        cb.set_label(label, fontsize = Title_size_axis)
        
        fig.savefig(f"Fig{figcode}{Title[i]}({system[i]}).png",dpi = 1000)
    
        # Diffusion final profile:

        fig2,ax2 = plt.subplots(1,1,figsize=(fsx,fsy))

        ax2.scatter(x,C[i][:,-1],label = f'{Title[i]} concentration final profile ({system[i]}, numerical)',color = color_seq[i])
        ax2.scatter(x,C[i][:,0],label = f'{Title[i]} concentration initial profile ({system[i]})',color = color_seq[i],alpha = 0.3)
        ax2.scatter(x,C_BT[i][:,-1],label = f'{Title[i]} concentration final profile ({system[i]}, analytical)',\
                    color = color_seq_BT[i])

        ax2.set_xlim(-10,xmax+dx)
        ax2.set_xticks(np.arange(0, xmax+dx, xmax/10))
        ax2.set_xlabel("Distance (x = 0 and x = 500 is the boundary)",fontsize = Title_size_axis)
        ax2.set_ylabel("Concentration (wt%)",fontsize = Title_size_axis)
        ax2.set_title(f"{Title[i]} ({system[i]})",fontsize = Title_size)
        ax2.tick_params(axis='both', which='major', labelsize=Ticks_size)
        ax2.text(0.65, 0.75, f'Initial concentration\nAl:{C0_Al}%\nCa:{C0_Ca}%\nNa:{C0_Na}%',\
                    transform=ax2.transAxes, fontsize=legend_size*0.75, color='black',\
                    bbox = dict(facecolor = 'white',edgecolor = 'black',alpha = 0.7))

        plt.grid(True)
        plt.legend(loc = 'lower left',fontsize=legend_size*0.75)
        fig2.savefig(f"Fig{figcode}{Title[i]}({system[i]})FP",dpi=1000)
        plt.close("all")
    
    figt,axest = plt.subplots(2,2,figsize=(fsx*4,fsy*2.5))

    # Maps figures
    img1 = mpimg.imread(f'Fig{figcode}Al(Diorite).png')
    axest[0,0].imshow(img1)
    axest[0,0].axis('off')
    img2 = mpimg.imread(f'Fig{figcode}Ca(Diorite).png')
    axest[0,1].imshow(img2)
    axest[0,1].axis('off')
    img3 = mpimg.imread(f'Fig{figcode}Na(Diorite).png')
    axest[1,0].imshow(img3)
    axest[1,0].axis('off')

    if basalts == True:
        img4 = mpimg.imread(f'Fig{figcode}Na(Basalts).png')
        axest[1,1].imshow(img4)
        axest[1,1].axis('off')
    else:
        img4 = mpimg.imread(f'Fig{figcode}Na(Diorite).png')
        axest[1,1].imshow(img4)
        axest[1,1].axis('off')       
    
    if basalts == False:
        figt.delaxes(axest[1, 1])

    figt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.1, hspace=0.1)

    # Add the number of subplots
    axest[0,0].text(0.03, 0.97, 'A', transform=axest[0,0].transAxes, fontsize=Title_size, fontweight='bold', color='black')
    axest[0,1].text(0.03, 0.97, 'B', transform=axest[0,1].transAxes, fontsize=Title_size, fontweight='bold', color='black')
    axest[1,0].text(0.03, 0.97, 'C', transform=axest[1,0].transAxes, fontsize=Title_size, fontweight='bold', color='black')

    if basalts == True:
        axest[1,1].text(0.03, 0.97, 'D', transform=axest[1,1].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    # figt.suptitle("The concentration maps for diffusion model",size = Title_size*2)
    
    figt.savefig(f"Fig{figcode}map.png",dpi = 1000)
    plt.close("all")


    # Final profile figures 
    figfp,axesfp = plt.subplots(2,2,figsize=(fsx,fsy))
    img1 = mpimg.imread(f'Fig{figcode}Al(Diorite)FP.png')
    axesfp[0,0].imshow(img1)
    axesfp[0,0].axis('off')
    img2 = mpimg.imread(f'Fig{figcode}Ca(Diorite)FP.png')
    axesfp[0,1].imshow(img2)
    axesfp[0,1].axis('off')
    img3 = mpimg.imread(f'Fig{figcode}Na(Diorite)FP.png')
    axesfp[1,0].imshow(img3)
    axesfp[1,0].axis('off')

    if basalts == True:
        img4 = mpimg.imread(f'Fig{figcode}Na(Basalts)FP.png')
        axesfp[1,1].imshow(img4)
        axesfp[1,1].axis('off')

    if basalts == False:
        figfp.delaxes(axesfp[1, 1])

    # Add the number of subplots
    axesfp[0,0].text(0.05, 0.95, 'A', transform=axesfp[0,0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axesfp[0,1].text(0.05, 0.95, 'B', transform=axesfp[0,1].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axesfp[1,0].text(0.05, 0.95, 'C', transform=axesfp[1,0].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    if basalts == True:
        axesfp[1,1].text(0.05, 0.95, 'D', transform=axesfp[1,1].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    # figfp.suptitle(f"Final concentration profiles (t={tmax}s)",size = Title_size)
    figfp.tight_layout()
    figfp.savefig(f"Fig{figcode}FP.png",dpi = 500)
    plt.close("all")

def Diffusionmap_NACS_Fig3():
    '''
    This function is to generate Fig. 3Map and Fig. 3FP (Na diffusion heatmap and profile in different system)

    '''
    # Figure parameters:
    fsx = 8 
    fsy = 6
    legend_size = 18
    psize = 6
    line_width = 3
    markerpattern = 'o'
    Title_size = 18
    Title_size_axis = 18
    Ticks_size = 18

    Diffusionmap_NACS(C0_Al = 9.5039, C0_Ca = 25.6461, C0_Na = 0.35, figcode = 3, basalts = True)
    figt,axest = plt.subplots(2,1,figsize=(fsx,fsy))
    figcode = 3
    img1 = mpimg.imread(f'Fig{figcode}Na(Diorite).png')
    axest[0].imshow(img1)
    axest[0].axis('off')
    img2 = mpimg.imread(f'Fig{figcode}Na(Basalts).png')
    axest[1].imshow(img2)
    axest[1].axis('off')

    figt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.1, hspace=0.01)

    # Add the number of subplots
    axest[0].text(0.03, 0.97, 'A', transform=axest[0].transAxes, fontsize=Title_size, fontweight='bold', color='black')
    axest[1].text(0.03, 0.97, 'B', transform=axest[1].transAxes, fontsize=Title_size, fontweight='bold', color='black')

    # figt.suptitle("The concentration maps for diffusion model",size = Title_size)
    
    figt.savefig(f"Fig{figcode}map.png",dpi = 1000)
    plt.close("all")

    figfp,axesfp = plt.subplots(1,2,figsize=(fsx,fsy))
    img1 = mpimg.imread(f'Fig{figcode}Na(Diorite)FP.png')
    axesfp[0].imshow(img1)
    axesfp[0].axis('off')
    img2 = mpimg.imread(f'Fig{figcode}Na(Basalts)FP.png')
    axesfp[1].imshow(img2)
    axesfp[1].axis('off')

    figfp.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.1, hspace=0.01)

    axesfp[0].text(0.05, 0.95, 'A', transform=axesfp[0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axesfp[1].text(0.05, 0.95, 'B', transform=axesfp[1].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    tmax = 500

    # figfp.suptitle(f"Final concentration profiles (t={tmax}s)",size = Title_size)
    figfp.tight_layout()
    figfp.savefig(f"Fig{figcode}FP.png",dpi = 500)
    plt.close("all")

def degassing_ingassing(T = 1640, C0wt = 0.35,M = MNa2O, Ncation = 2, V_surrounding = 6.545*(10**7), a = -9.251, b = -19628,\
                        xmax = 300,tmax = 100,dx = 5,dt = 0.008,\
                        constant_boundary = True,debug = False, C_surrounding = 0,speed = 1,\
                        Temperature_decrease = False, k =60,\
                        exp = False, a0 = 0.05, b0 = 1183):
    '''
    This function is to give the numerical solution of Na in basaltic system with equal-volume vaccum boundary condition.

    ----------
    Parameters
    ----------

    T: float, defaults is 1640
        temperature, unit is K
    C0wt: float, defaults is 0.35
        initial concentration of Na2O, unit is %
    M: float, defaults is MNa2O
        the mole mass of Na2O
    Ncation: int, defaults is 2
        the number of cations
    V_surrounding: float, defaults is 6.545*10^7
        the volume of surrounding, unit is um^3
    a: float, defaults is -9.251
        the a parameter of Na diffusion coefficient 
    b: float, defaults is -19628
        the b parameter of Na diffusion coefficient 
    xmax: float, defaults is 300
        the diameter of the glass bead, unit is um
    tmax: float, defaults is 100
        the time when diffusion end, unit is seconds
    dx: float, defaults is 5
        the intervals between two nearby x grid, unit is um
    dt: float, defaults is 0.008
        the intervals between two nearby t grid, unit is seconds
    constant_boundary: bool, defaults is True
        the boundary concentration is constant if this is True
    debug: bool, defualts is False,
        the debug process
    C_surrounding: float, default is 0
        initial surrounding concentration
    speed: float, default is 1
        describe how fast the liner growth of surrounding concentration
    Temperature_decrease: bool, default is False
        If this true, we will include the variation of temperature
    k: float, default is 60
        One of the parameter in temperature decrease
    exp: bool, default is False, 
        If this true, we will use the exponential form of growth for surrounding concentration
    a0: float, default is 0.05
        parameter a of surrounding concentration increase 
    b0: float, default is 1183
        parameter b of surrounding concentration increase 
    
    ----------
    Return
    ----------

    xgrid: float array
        a series of points on the diameter of the glass bead, unit is um
    tgrid: float array
        a series of time points, unit is seconds
    C: float array
        concentration array of Na in x-t space, units are mol/um^3
    tmax: float
        the time when diffusion end, unit is seconds 
    C0: float
        initial concentration
    
    '''
    xgrid, tgrid = np.arange(0,xmax+dx,dx),np.arange(0,tmax+dt,dt)
    M = int(np.round(xmax/dx+1))
    N = int(np.round(tmax/dt+1))
    C = np.zeros((M,N))
    N0 = C0wt/M*Ncation # The initial mole number of cation in glass bead
    C0 = (C0wt/M*Ncation)/((xmax/2)**3*4/3*3.14) # The initial concentration of cation in glass bead, unit: mol/um3
    k = 60 # The time when temperature cools to half of the initial temperature, unit: s

    # Set initial condition

    for i in range(M):
        C[i,0] = C0
    C[0,0] = 0
    C[M-1,0] = 0
    N_inner = N0
    N_surrounding = 0
    T0 = T

    if debug:
        t_path = np.zeros(N)
        N_surr_path = np.zeros(N)
        C_surr_path = np.zeros(N)
        N_inner_path = np.zeros(N)
        C_inner_path = np.zeros(N)
        N_inner_path[0] = N_inner
        C_inner_path[0] = C0

    for j in range(0,N-1):

        # Set Diffusivity
        D = (10**12)*np.exp(a+b/T) # unit um2/s

        for i in range(1,M-1):
            C[i,j+1] = dt*D*(C[i+1,j]-2*C[i,j]+C[i-1,j])/(dx**2)+C[i,j]

        if constant_boundary:
            C[0,:] = C_surrounding
            C[M-1,:] = C_surrounding
        else:
            # Set boundary condition
            N_surrounding = N_surrounding+(N_inner-(np.mean(C[1:-1,j+1]))*((xmax/2)**3)*4/3*3.14)
            
            C_surrounding = N_surrounding/V_surrounding

            C[0,j+1] = C_surrounding
            C[M-1,j+1] = C_surrounding
            N_inner = np.mean(C[1:-1,j+1])*((xmax/2)**3)*4/3*3.14
        if Temperature_decrease:
            T = T0/(1+((j+1)*dt)/k)

        if debug:
            t_path[j+1] = j
            N_surr_path[j+1] = N_surrounding
            C_surr_path[j+1] = C_surrounding
            C_inner_path[j+1] = np.mean(C[1:-1,j])
            N_inner_path[j+1] = N_inner

    if debug:

        figd= plt.figure(figsize=(8,6))
        axd = figd.add_subplot(111)
        axd.scatter(t_path,C_surr_path,color = 'red')
        axd.scatter(t_path,C_inner_path,color = 'blue')
        print(C_inner_path)
        plt.show()
        print("This is initial concentration:",C0)
        print("This is the first time point of C:" ,C[:,0])
        print("This is the second time point of C:" ,C[:,1])
        print("This is the final time point of C:", C[:,N-1])
    
    return xgrid, tgrid, C, tmax, C0

def Increase_boundary(T = 1640, C0wt = 0.35,M = MNa2O, Ncation = 2, V_surrounding = 6.545*(10**7), a = -9.251, b = -19628,\
                        xmax = 300,tmax = 100,dx = 5,dt = 0.008,\
                        constant_boundary = True,debug = False, C_surrounding = 0,speed = 1,\
                         Temperature_decrease = False, k = 60, \
                        exp = False, a0 = 0.05, b0 = 1183):
    '''
    This function is to give the numerical solution of Na in basaltic system with increase boundary.

    ----------
    Parameters
    ----------

    T: float, defaults is 1640
        temperature, unit is K
    C0wt: float, defaults is 0.35
        initial concentration of Na2O, unit is %
    M: float, defaults is MNa2O
        the mole mass of Na2O
    Ncation: int, defaults is 2
        the number of cations
    V_surrounding: float, defaults is 6.545*10^7
        the volume of surrounding, unit is um^3
    a: float, defaults is -9.251
        the a parameter of Na diffusion coefficient 
    b: float, defaults is -19628
        the b parameter of Na diffusion coefficient 
    xmax: float, defaults is 300
        the diameter of the glass bead, unit is um
    tmax: float, defaults is 100
        the time when diffusion end, unit is seconds
    dx: float, defaults is 5
        the intervals between two nearby x grid, unit is um
    dt: float, defaults is 0.008
        the intervals between two nearby t grid, unit is seconds
    constant_boundary: bool, defaults is True
        the boundary concentration is constant if this is True
    debug: bool, defualts is False,
        the debug process
    C_surrounding: float, default is 0
        initial surrounding concentration
    speed: float, default is 1
        describe how fast the liner growth of surrounding concentration
    Temperature_decrease: bool, default is False
        If this true, we will include the variation of temperature
    k: float, default is 60
        One of the parameter in temperature decrease
    exp: bool, default is False, 
        If this true, we will use the exponential form of growth for surrounding concentration
    a0: float, default is 0.05
        parameter a of surrounding concentration increase 
    b0: float, default is 1183
        parameter b of surrounding concentration increase 
    
    ----------
    Return
    ----------

    xgrid: float array
        a series of points on the diameter of the glass bead, unit is um
    tgrid: float array
        a series of time points, unit is seconds
    C: float array
        concentration array of Na in x-t space, units are mol/um^3
    tmax: float
        the time when diffusion end, unit is seconds 
    C0: float
        initial concentration
    
    '''
        
    xgrid, tgrid = np.arange(0,xmax+dx,dx),np.arange(0,tmax+dt,dt)
    M = int(np.round(xmax/dx+1))
    N = int(np.round(tmax/dt+1))
    C = np.zeros((M,N))
    C0 = (C0wt/M*Ncation)/((xmax/2)**3*4/3*3.14) # The initial concentration of cation in glass bead, unit: mol/um3
    T0 = T

    # Set initial condition

    for i in range(M):
        C[i,0] = C0
    C[0,0] = 0
    C[M-1,0] = 0

    if exp:
        C[0,0] = a0*np.exp(b0/T)
        C[M-1,0] = a0*np.exp(b0/T)

    for j in range(0,N-1):

        # Set Diffusivity
        D = (10**12)*np.exp(a+b/T) # unit um2/s

        for i in range(1,M-1):
            C[i,j+1] = dt*D*(C[i+1,j]-2*C[i,j]+C[i-1,j])/(dx**2)+C[i,j]
        
        if Temperature_decrease:
            T = T0/(1+((j+1)*dt)/k)
        
        if constant_boundary:
            C[0,:] = C_surrounding
            C[M-1,:] = C_surrounding
        else:
            # Set boundary condition 
            if exp:
                C_surrounding = a0*np.exp(b0/T)
                speed = 'Exp'
            else:         
                C_surrounding = C_surrounding+speed*2.09816411*10**(-10)*dt/tmax
            C[0,j+1] = C_surrounding
            C[M-1,j+1] = C_surrounding
        

    
    return xgrid, tgrid, C, tmax, C0

def gas_map(T = 1640, C0wt = 0.35,M = MNa2O, Ncation = 2, V_surrounding = 6.545*10**7, \
            a = -9.251, b = -19628,\
            xmax = 300,tmax = 50,dx = 5,dt = 0.008,constant_boundary = False,\
            Title = 'Na',system = 'Basalts',figcode = 4,debug = False,\
            C_surrounding = 0,model = degassing_ingassing, speed = 1, only_half = False, half_initial = False,\
            Temperature_decrease = False, k = 60,\
            exp = False, a0 = 0.05, b0 = 1183):
    '''
    This function is to generate the heatmap and concentration profiles at different time of Na.
    ----------
    Parameters
    ----------

    T: float, defaults is 1640
        temperature, unit is K
    C0wt: float, defaults is 0.35
        initial concentration of Na2O, unit is %
    M: float, defaults is MNa2O
        the mole mass of Na2O
    Ncation: int, defaults is 2
        the number of cations
    V_surrounding: float, defaults is 6.545*10^7
        the volume of surrounding, unit is um^3
    a: float, defaults is -9.251
        the a parameter of Na diffusion coefficient 
    b: float, defaults is -19628
        the b parameter of Na diffusion coefficient 
    xmax: float, defaults is 300
        the diameter of the glass bead, unit is um
    tmax: float, defaults is 100
        the time when diffusion end, unit is seconds
    dx: float, defaults is 5
        the intervals between two nearby x grid, unit is um
    dt: float, defaults is 0.008
        the intervals between two nearby t grid, unit is seconds
    constant_boundary: bool, defaults is False
        the boundary concentration is constant if this is True
    Title: char, defaults is 'Na'
        the name of the element we study
    system: char, defaults is 'Basalts'
        the name of the system we study
    figcode: int, defaults is 4
        the number of our figure
    debug: bool, defualts is False,
        the debug process
    C_surrounding: float, default is 0
        initial surrounding concentration
    model: func, defaults is degassing_ingassing
        decide which model we are going to draw
    speed: float, default is 1
        describe how fast the liner growth of surrounding concentration
    only_half: bool, defaults is False
        if it is true, we only draw the half time profile
    half_initial: bool, defaults is False
        if it is true, we draw both the half time profile and initial profile
    Temperature_decrease: bool, defaults is False
        If this true, we will include the variation of temperature
    k: float, default is 60
        One of the parameter in temperature decrease
    exp: bool, default is False, 
        If this true, we will use the exponential form of growth for surrounding concentration
    a0: float, default is 0.05
        parameter a of surrounding concentration increase 
    b0: float, default is 1183
        parameter b of surrounding concentration increase 
    
    ----------
    Return
    ----------

    xgrid: float array
        a series of points on the diameter of the glass bead, unit is um
    tgrid: float array
        a series of time points, unit is seconds
    C: float array
        concentration array of Na in x-t space, units are mol/um^3
    tmax: float
        the time when diffusion end, unit is seconds 
    C0: float
        initial concentration
    
    '''
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

    x,time,C,tmax,C0 = model(T = T, C0wt = C0wt, M = M, Ncation = Ncation, V_surrounding = V_surrounding, \
                                        a = a, b = b,\
                                        xmax = xmax,tmax = tmax,dx = dx,dt = dt,\
                                        C_surrounding = C_surrounding,speed = speed,\
                                        constant_boundary = constant_boundary,\
                                        Temperature_decrease = Temperature_decrease,k=k,\
                                        exp = exp, a0 = a0, b0 = b0,\
                                        debug = debug)
    if Temperature_decrease:
        temp_inform = 'Decrease'
    else:
        temp_inform = 'Constant'
    
    if only_half or half_initial:
        print("No concentration map")
    else:
        # Create a figure/axes object
        fig,ax= plt.subplots(1,1,figsize=(fsx,fsy))

        # Create a color map and add a color bar.
        map = ax.pcolor(time, x, C, cmap='magma', vmin=0, vmax=C0, shading='nearest')

        ax.set_title(f"The {Title} diffusion model",fontsize = Title_size)
        ax.set_xlabel("Time (s)", fontsize = Title_size_axis)
        ax.set_ylabel("Distance (um)", fontsize = Title_size_axis)
        ax.tick_params(axis='both', which='major', labelsize = Ticks_size)
        ax.set_xlim(0,tmax+dt)
        ax.set_xticks(np.arange(0,tmax+dt,tmax/10))
        ax.text(0.05, 0.6, f'T0 = {T}, speed = {speed}, Temperature:{temp_inform}',\
                transform=ax.transAxes, fontsize=legend_size-2, color='black',\
                bbox = dict(facecolor = 'white',edgecolor = 'lightgray',alpha = 0.6))

        label = 'Concentration(mol/um^3)'

        cb = plt.colorbar(map, ax=ax, label=label)
        cb.ax.tick_params(labelsize = Ticks_size)
        cb.set_label(label, fontsize = legend_size)
        fig.savefig(f"Fig{figcode}map.png",dpi = 500)

    
    fig2,ax2 = plt.subplots(1,1,figsize=(fsx,fsy))

    ax2.scatter(x,C[:,int(len(time)/2)-1],label = f'Time = {tmax/2}s',color = 'red')
    if only_half:
        print(f"Only show the profile of {tmax/4}s")
    elif half_initial:
        ax2.scatter(x,C[:,0],label = f'Time = 0s',color = 'black')
    else:
        ax2.scatter(x,C[:,int(len(time)/4)-1],label = f'Time = {tmax/4}s',color = 'orange')
        ax2.scatter(x,C[:,int(len(time))-1],label = f'Time = {tmax}s',color = 'blue')
        ax2.scatter(x,C[:,0],label = f'Time = 0s',color = 'black')

    ax2.set_xlim(-10,xmax+dx)
    ax2.set_xticks(np.arange(0, xmax+dx, xmax/10))
    ax2.set_xlabel(f"Distance (x = 0 and x = {xmax} is the boundary)",fontsize = Title_size_axis)
    ax2.set_ylabel("Concentration (mol/um^3)",fontsize = Title_size_axis)
    ax2.tick_params(axis='both', which='major', labelsize=Ticks_size)
    ax2.set_title(f"{Title} concentration profile",size = Title_size)
    ax2.text(0.2, 0.6, f'T0 = {T}, speed = {speed}, Temperature:{temp_inform}',\
                transform=ax2.transAxes, fontsize=legend_size-2, color='black',\
                bbox = dict(facecolor = 'white',edgecolor = 'lightgray',alpha = 0.6))

    ax2.grid(True)
    ax2.legend(loc = 'best',fontsize = legend_size)
    fig2.savefig(f"Fig{figcode}FP",dpi=500)
    plt.close("all")

def Na_diff_vary_boundary():
    '''
    This function is to generate the heatmap and concentration profiles 
    at different time of Na with equal-volume vaccum boundary condition
    '''
    gas_map()

def Na_diff_increase_boundary():
    '''
    This function is to generate the heatmap and concentration profiles 
    at different time of Na with liner growth concentration (x1) boundary condition
    '''

    gas_map(figcode = 5, model = Increase_boundary)

def Na_diff_increase_boundary_speed_up():
    '''
    This function is to generate the heatmap and concentration profiles 
    at different time of Na with liner growth concentration (x3) boundary condition
    '''
    gas_map(figcode = 6, model = Increase_boundary, speed = 3)
    gas_map(figcode = 7, model = Increase_boundary, speed = 3, only_half = True)

def Na_diff_vary_boundary_temperature_decrease():
    '''
    This function is to generate the heatmap and concentration profiles 
    at different time of Na with equal-volume vaccum boundary condition and decrease temperature
    '''
    gas_map(T = 1640, figcode = 8, model = degassing_ingassing, speed = 1,Temperature_decrease = True,tmax = 200)

def Na_diff_increase_boundary_temperature_decrease():

    '''
    This function is to generate the heatmap and concentration profiles 
    at different time of Na with liner growth concentration (x10, x50, x100) boundary conditions and decrease temperature
    '''

    gas_map(T = 1640, figcode = '10A', model = Increase_boundary, speed = 1,Temperature_decrease = True,tmax = 50)
    gas_map(T = 1640, figcode = '10B', model = Increase_boundary, speed = 10,Temperature_decrease = True,tmax = 50)
    gas_map(T = 1640, figcode = '10C', model = Increase_boundary, speed = 50,Temperature_decrease = True,tmax = 50)
    gas_map(T = 1640, figcode = '10D', model = Increase_boundary, speed = 100,Temperature_decrease = True,tmax = 50)

    fsx = 16
    fsy = 12
    legend_size = 25

    fig,axes = plt.subplots(2,2,figsize = (16,12))

    # Import subplots
    img1 = mpimg.imread(f'Fig10AFP.png')
    axes[0,0].imshow(img1)
    axes[0,0].axis('off')

    img2 = mpimg.imread(f'Fig10BFP.png')
    axes[0,1].imshow(img2)
    axes[0,1].axis('off')

    img3 = mpimg.imread(f'Fig10CFP.png')
    axes[1,0].imshow(img3)
    axes[1,0].axis('off')

    img4 = mpimg.imread(f'Fig10DFP.png')
    axes[1,1].imshow(img4)
    axes[1,1].axis('off')


    # Add the number of subplots
    axes[0,0].text(0.05, 0.95, 'A', transform=axes[0,0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axes[0,1].text(0.05, 0.95, 'B', transform=axes[0,1].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axes[1,0].text(0.05, 0.95, 'C', transform=axes[1,0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axes[1,1].text(0.05, 0.95, 'D', transform=axes[1,1].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    # Generate the final figure and save
    plt.tight_layout()
    fig.savefig(f"Fig10FP.png",dpi = 500)
    plt.close("all")

    # Import subplots
    img1 = mpimg.imread(f'Fig10Amap.png')
    axes[0,0].imshow(img1)
    axes[0,0].axis('off')

    img2 = mpimg.imread(f'Fig10Bmap.png')
    axes[0,1].imshow(img2)
    axes[0,1].axis('off')

    img3 = mpimg.imread(f'Fig10Cmap.png')
    axes[1,0].imshow(img3)
    axes[1,0].axis('off')

    img4 = mpimg.imread(f'Fig10Dmap.png')
    axes[1,1].imshow(img4)
    axes[1,1].axis('off')


    # Add the number of subplots
    axes[0,0].text(0.05, 0.95, 'A', transform=axes[0,0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axes[0,1].text(0.05, 0.95, 'B', transform=axes[0,1].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axes[1,0].text(0.05, 0.95, 'C', transform=axes[1,0].transAxes, fontsize=legend_size, fontweight='bold', color='black')
    axes[1,1].text(0.05, 0.95, 'D', transform=axes[1,1].transAxes, fontsize=legend_size, fontweight='bold', color='black')

    # Generate the final figure and save
    plt.tight_layout()
    fig.savefig(f"Fig10map.png",dpi = 500)
    plt.close("all")

def Na_diff_increase_boundary_exp_temperature_decrease():
    '''
    This function is to generate the heatmap and concentration profiles 
    at different time of Na with exponential increase boundary condition and decrease temperature

    '''
    gas_map(T = 1640, figcode = 11,tmax = 150, model = Increase_boundary,Temperature_decrease = True, k =180,\
            exp = True,a0 = 3.31*10**(-10),b0 = 1183)
    # gas_map(T = 1640, figcode = 11,tmax = 150, model = Increase_boundary,Temperature_decrease = True, k =180,\
    #         exp = True,a0 = 3.31*10**(-10),b0 = 1183,half_initial = True)


