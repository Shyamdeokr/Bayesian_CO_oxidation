#!/usr/bin/env python
# coding: utf-8

# # Microkinetic Model for CO Oxidation (Pd), (PdO) and (PdO2) + O2 ads

# Reactions - 1 - 8
# 
# ![image.png](attachment:image.png)

# Reactions 6' - 11
# ![image.png](attachment:image.png)

# Reactions 12-14
# ![image.png](attachment:image.png)

# Reactions 15-16 for O2 adsorption on Pd
# 
# ![image.png](attachment:image.png)

# First, we import the necessary numpy and scipy modules

# In[1]:


import numpy as np
import math
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# We also need to define a set of reaction conditions

# # Microkinetic Model

# In[2]:


# T    = 480      # K
# PCO  = 0.02     # bar PCO is 2%
# PO2  = 0.20     # bar PO2 is 20%
PCO2 = 0        # bar
R = 8.314
mAr = 39.848
mCO = 28.0101
mO2 = 32
mCO2 = 44

SCOn = 197.660
SO2n = 205.152
SCO2n = 213.79


# ... and a few physical constants and conversion factors

# In[3]:


J2eV = 6.24150974E18            # eV/J
Na   = 6.0221415E23             # mol-1
h    = 6.626068E-34 * J2eV      # in eV*s
kb   = 1.3806503E-23 * J2eV     # in eV/K
# kbT  = kb * T                   # in eV


# ## Rate constants

# In[4]:


def get_rate_constants(T):
    kbT  = kb * T                   # in eV
    # Gas phase entropies converted to eV/K
    SCOg  = 197.66 * J2eV / Na          # eV/K
    SO2g  = 205.0 * J2eV / Na
    SCO2g = 213.74 * J2eV / Na
   
    # Surface entropies as per charlie campbell's paper
#   SCO2v = Sv = SO2v = SO = SCOO = 0
    SCO = (0.70*SCOn - 3.3*R)*J2eV/Na
    SCO2 = (0.70*SCO2n - 3.3*R)*J2eV/Na
    SO2 = (0.70*SO2n - 3.3*R)*J2eV/Na
    
    # Reaction energies
    dE = np.zeros(19)             # array initialization
    dE[0] = -1.09                 # CO adsorption (-1.09)
    dE[1] = -0.22                 # 1st CO oxidation
    dE[2] =  0.52                 # 1st CO2 desorption (0.52)
    dE[3] = -2.07                 # O2 adsorption in a vacancy
    dE[4] = -1.29                 # O2 dissociation in the vacancy
    dE[5] = -0.65                 # 2nd CO adsorption (-0.65)
    dE[6] = -1.46                 # 2nd CO oxidation
    dE[7] = -0.0057               # 2nd CO2 desorption
    dE[8] = -0.65                 # rxn 6'- 2nd CO adsorption using lattice O
    dE[9] =  0.33                 # rxn 7' - 2nd CO oxidation using lattice O (0.33)
    dE[10] = 0.58                 # rxn 8' - 2nd CO2 desorption leading to (V+O)*
    dE[11] = -2.34                # rxn 9 - O2 adsorption at (v+O)*
    dE[12] = -0.97                # rxn 10 -O2 dissociation at (v+O)*
    dE[13] = -1.27                # rxn 11 - O migration from (V+O)* --> *
    dE[14] = -0.35                # rxn 12 - CO adsorption at 2O*
    dE[15] = -2.31                # rxn 13 - CO oxidation at 2O*
    dE[16] = 0.45                # rxn 14 - CO2 desorption from O*
    dE[17] = -0.0067              # rxn 15 - O2 adsorption on *
    dE[18] = -1.77                # rxn 16 - O2 dissociation on * to give 2O*



    # Entropy changes (Ignoring dependence on T)
#     dSS = 0.001
    dS = np.zeros(19)                    # array initialization
    dS[0] = SCO - SCOg                  # CO adsorption
    dS[1] = 0                           # 1st CO oxidation
    dS[2] = SCO2g - SCO2                # 1st CO2 desorption
    dS[3] = SO2 - SO2g                  # O2 adsorption in a vacancy
    dS[4] = 0                           # O2 dissociation in the vacancy
    dS[5] = SCO - SCOg                  # 2nd CO adsorption
    dS[6] = 0                           # 2nd CO oxidation
    dS[7] = SCO2g - SCO2                # 2nd CO2 desorption
    dS[8] = SCO - SCOg                  # rxn 6'- 2nd CO adsorption using lattice O
    dS[9] = 0                           # rxn 7' - 2nd CO oxidation using lattice O
    dS[10] = SCO2g - SCO2               # rxn 8' - 2nd CO2 desorption leading to (V+O)*
    dS[11] = SO2 - SO2g                 # rxn 9 - O2 adsorption at (v+O)*
    dS[12] = 0                          # rxn 10 -O2 dissociation at (v+O)*
    dS[13] = 0                          # rxn 11 - O migration from (V+O)* --> *
    dS[14] = SCO - SCOg                 # rxn 12 - CO adsorption at 2O*
    dS[15] = 0                          # rxn 13 - CO oxidation at 2O*
    dS[16] = SCO2g - SCO2               # rxn 14 - CO2 desorption from O*
    dS[17] = SO2 - SO2g                 # rxn 15 - O2 adsorption on *
    dS[18] = 0                          # rxn 16 - O2 dissociation on * to give 2O*

    
    # Activation energy barriers
    Ea = np.zeros(19)              # array initialization
    Ea[1] =  0.49                 # 1st CO Oxidation barrier = 0.49
    Ea[9] =  0.69                 # 2nd CO Oxidation barrier using lattice O (0.69)
    Ea[12] = 0.51                 # O2 dissociation barrier at (v+O)*
    Ea[13] = 0.18                 # O migration barrier from (v+O)* ---> *
    Ea[18] = 1.44                 # rxn 16 - O2 dissociation on * to give 2O*
    Ea[6] =  0.19                 # 2nd CO Oxidation barrier (assumed)
    

    # Entropy changes to the transition state
    STS = np.zeros(19)             # array initialization                  
    STS[0] =  (0.30*SCOn/R + 3.3-1/3*(18.6+math.log((mCO/mAr)**1.5*(T/298)**2.5)))*R      # 1st CO adsorption entropy - ignoring 
    # SCOn dependence on T
    STS[2] =  (0.30*SCO2n/R + 3.3-1/3*(18.6+math.log((mCO2/mAr)**1.5*(T/298)**2.5)))*kb      # 1st CO2 adsorption entropy           
    STS[3] =  (0.30*SO2n/R + 3.3-1/3*(18.6+math.log((mO2/mAr)**1.5*(T/298)**2.5)))*R      # O2 adsorption entropy    
    STS[5] =  STS[14]= STS[8] = STS[0]      # CO adsorption entropiesSTS[7] =  STS[2]   
    STS[10]= STS[16] = STS[2]   # CO2 adsorption entropies
    STS[17]= STS[11] = STS[3]

    # Calculate equilibrium and rate constants
    K  = [0]*19                   # equilibrium constants
    kf = [0]*19                   # forward rate constants
    kr = [0]*19                   # reverse rate constants
    for i in range(19):
        dG = dE[i] - T*dS[i]
        K[i]  = np.exp(-dG/kbT)
        
        # Enforce Ea > 0, and Ea > dE 
        if i not in [0,3,5,8,11,14,17]: #(steps 0, 3 and 5 are adsorption steps)
            Ea[i] = max([0,dE[i],Ea[i]])   
            kf[i] = kbT/h * np.exp(STS[i]/kb) * np.exp(-Ea[i]/kbT)
            kr[i] = kf[i]/K[i] # enforce thermodynamic consistency
        else:
            Ea[i] =-dE[i]                                         # Ea[i] = Eads
            kr[i] = kbT/h * np.exp(STS[i]/R) * np.exp(-Ea[i]/kbT) # STS = TS-ads for adsorption 0,3 and 5 
            kf[i] = K[i]*kr[i]
            
    return (kf,kr,Ea)  


# ## Rates and intermediate species coverage (theta)

# Next, we need to calculate the rates for each step. As input we provide the rate constants $k_i$ and the coverages $\theta_i$. The input variables are passed as arrays.

# In[5]:


def get_rates(theta,kf,kr,P):
    
    # Extract elements of theta and assign them
    # to more meaningful variables
    tCO   = theta[0]              # coverage of CO
    tCO2v   = theta[1]              
    tvac    = theta[2]  
    tO2v = theta[3]
    tO = theta[4]
    tCOO = theta[5]
    tCO2 = theta[6]
    tCOOL = theta[7]
    tCO2L = theta[8]
    tvplusO = theta[9]
    tO2plusvplusO = theta[10]
    t2O = theta[11]
    tCOplus2O = theta[12]
    tCO2plusO = theta[13]
    tO2 = theta[14]
    tstar = 1.0 - tCO - tCO2v - tvac - tO2v - tO - tCOO - tCO2 - tCOOL -tCO2L - tvplusO -tO2plusvplusO -t2O -tCOplus2O -tCO2plusO - tO2  # site balance
    
    PCO = P[0]
    PO2 = P[1]

    # Caluclate the rates: eqns (39)-(42)
    rate    = [0]*19               # array with 8 zeros
    rate[0] = kf[0] * PCO * tstar - kr[0] * tCO
    rate[1] = kf[1] * tCO - kr[1] * tCO2v
    rate[2] = kf[2] * tCO2v - kr[2] * PCO2 * tvac
    rate[3] = kf[3] * tvac * PO2 - kr[3] * tO2v
    rate[4] = kf[4] * tO2v - kr[4] * tO
    rate[5] = kf[5] * tO * PCO - kr[4] * tCOO
    rate[6] = kf[6] * tCOO - kr[6] * tCO2
    rate[7] = kf[7] * tCO2 - kr[7] * PCO2 * tstar
    
    rate[8] = kf[8] * tO * PCO - kr[8] * tCOOL
    rate[9] = kf[9] * tCOOL - kr[9] * tCO2L
    rate[10] = kf[10] * tCO2L - kr[10] * PCO2 * tvplusO
    
    rate[11] = kf[11] * tvplusO * PO2 - kr[11] * tO2plusvplusO
    rate[12] = kf[12] * tO2plusvplusO - kr[12] * t2O
    rate[13] = kf[13] * tvplusO - kr[13] * tstar
    
    rate[14] = kf[14] * t2O * PCO - kr[14] * tCOplus2O
    rate[15] = kf[15] * tCOplus2O - kr[15] * tCO2plusO
    rate[16] = kf[16] * tCO2plusO - kr[16] * PCO2 * tO
    
    rate[17] = kf[17] * tstar * PO2 - kr[17] * tO2
    rate[18] = kf[18] * tO2 - kr[18] * t2O
    
    return rate 


# ## Solving ODE equations

# We also need to define the systems of ordinary differential equations (ODEs) that we intend to solve. Note that we solve the transient problem without assuming steady-state and integrate for very long times. This is often a more robust technique to find a physical solution.

# In[6]:


def get_odes(theta,t,kf,kr,P):
# returns the system of ODEs d(theta)/dt, calculated at the current value of theta (and time t)

    rate = get_rates(theta,kf,kr,P)       # calculate rates at current value of theta

    # Time derivatives of theta
    dt  = [0]*15
    dt[0] = rate[0] - rate[1]           # d(tCO)/dt
    dt[1] = rate[1] - rate[2]           
    dt[2] = rate[2] - rate[3]  
    dt[3] = rate[3] - rate[4] 
    dt[4] = rate[4] - rate[5] + rate[16]  - rate[8]       #d(tO)/dt
    dt[5] = rate[5] - rate[6] 
    dt[6] = rate[6] - rate[7] 
    
    dt[7] = rate[8] - rate[9]
    dt[8] = rate[9] - rate[10]
    
    dt[9] = rate[10] - rate[11] -rate[13]  #d(v+O)/dt
    
    dt[10] = rate[11] - rate[12]
    
    dt[11] = rate[12] - rate[14] + rate[18] #d(t2O)/dt
    
    dt[12] = rate[14] - rate[15]
    dt[13] = rate[15] - rate[16]
    dt[14] = rate[17] - rate[18]

    
    return dt


# To solve the system of ODEs we need to provide an initial guess. If we don't know any better, a good starting point is always a completely empty surface.

# In[7]:


# theta0 = (0.0, 0., 0., 0 , 0 , 0 , 0.0) # initial coverage of CO*, CO2+v*, vac*, O2v*, O*, COO*, CO2*, COO(L)*,CO2(L)*, (v+O)*,(O2+v+O)*, 2O*, (CO+2O)*, (CO2+O)*, O2* respectively
theta0 =np.zeros(15)


# And now we are ready to start the integration from $\theta_0$ to $\theta_{steady-state}$ over a long enough time span. Here, the time span is set to 1E6 sec, but you need to test if this is sufficient to reach steady-state. If you lower the time span and still obtain identical results, steady-state has been reached.

# We can now inspect the results. Since we may want to do this more than once, we can define a customized print function for our output.

# In[8]:


def solve_ode(kf,kr,theta0,P):
# Solve the system of ODEs using scipy.integrate.odeint
# Assumes an empty surface as initial guess if nothing else is provided
    from scipy.integrate import odeint

    # Integrate the ODEs for 1E6 sec (enough to reach steady-state)
    theta = odeint(get_odes,        # system of ODEs
                   theta0,          # initial guess
                   [0,1E6],         # time span
                   args = (kf,kr,P),  # arguments to get_odes()
                   h0 = 1E-36,      # initial time step
                   mxstep = 90000)  # maximum number of steps
#                    rtol = 1E-12,    # relative tolerance
#                    atol = 1E-15)    # absolute tolerance

    return theta[-1,:]


# ## Solutions to the rates and coverage of the intermediate species

# In[9]:


def print_output(theta0,T,P):
# Prints the solution of the model
    (kf,kr,Ea) = get_rate_constants(T)
    theta = solve_ode(kf,kr,theta0,P)
    rates   = get_rates(theta,kf,kr,P)
#     print ("the result is:")
#     print
#     for r,rate in enumerate(rates):
#         if r in [0,3,5,8,11,14,17]:
#             print ("Step",r,": rate =",rate,", kf =",kf[r],", kr=",kr[r],", reverse Ea =",Ea[r])
#         else:
#             print("Step",r,": rate =",rate,", kf =",kf[r],", kr=",kr[r],", Ea =",Ea[r])
#     print ("The coverages for CO*, CO2+v*, vac*, O2v*, O*, COO*, CO2*, COO(L)*,CO2(L)*, (v+O)*,(O2+v+O)*, 2O*, (CO+2O)*, (CO2+O)*, O2* are:")
#     for t in theta:
#         print (t)
    return (rates[1]+rates[6]+rates[9]+rates[15],theta[0])


# And we call the function with our output values, i.e., the last row of the result matrix $\theta$.

# In[10]:


print_output(theta0,473,[0.02,0.20])


# ## Reaction orders ($P_{CO}$ & $P_{O2}$) and apparent barrier

# In[11]:


def rxn_order_CO(T,P):
    gridpoints = 3
    rate_PCO=np.zeros(gridpoints)
    PCO1=P[0]
    PCO2=PCO1+0.05
    PCO_range = np.linspace(PCO1,PCO2,gridpoints)
    for i,PCO in enumerate(PCO_range):
        rate_PCO[i]=print_output(theta0,T,[PCO,P[1]])[0] # PO2 = 0.20 
        if rate_PCO[i]<10**-323:
            rate_PCO[i]=10**-323
    PCO_range = PCO_range.reshape(-1, 1)
    LR_CO=model.fit(np.log(PCO_range), np.log(rate_PCO))
    order_CO=LR_CO.coef_    # LR.intercept_
    return order_CO[0]


# In[12]:


def rxn_order_O2(T,P):
    gridpoints = 3
    rate_PO2=np.zeros(gridpoints)
    PO21=P[1]
    PO22=PO21+0.1
    PO2_range = np.linspace(PO21,PO22,gridpoints)
    for i,PO2 in enumerate(PO2_range):
        rate_PO2[i]=print_output(theta0,T,[P[0],PO2])[0] #PCO = 0.02
        if rate_PO2[i]<10**-323:
            rate_PO2[i]=10**-323
    PO2_range = PO2_range.reshape(-1, 1)
    LR_O2=model.fit(np.log(PO2_range), np.log(rate_PO2))
    order_O2=LR_O2.coef_ # LR.intercept_
    return order_O2[0]


# ## Apparent barrier

# In[13]:


def apparent_barrier(T,P):
    gridpoints = 3
    rate_app=np.zeros(gridpoints)
    T1=T-1
    T2=T+1
    T_range = np.linspace(T1,T2,gridpoints)
    for i,T in enumerate(T_range):
        rate_app[i]=print_output(theta0,T,P)[0] # PCO and PO2 are 0.02 and 0.20 respectively
        if rate_app[i]<10**-323:
            rate_app[i]=10**-323
        #     plt.plot(1/T_range, np.log(rate_app), 'ro')
#     plt.xlabel('1/T')
#     plt.ylabel('log[rate_CO2_formation]')
#     plt.show()
    T_range = T_range.reshape(-1, 1)
    LR=model.fit(1/T_range, np.log(rate_app))  #Arhenius equation
    LR.coef_
    # apparent_barrier=-LR.coef_*0.02568/298  #apparent barrier in eV
    apparent_barrier=-LR.coef_*kb  #apparent barrier in eV
    apparent_barrier[0]     # LR.intercept_
    return apparent_barrier[0]


# In[14]:


print('Reaction order of CO: %.4f ' % rxn_order_CO(473,[0.02,0.20]))
print('Reaction order of O2: %.4f ' % rxn_order_O2(473,[0.02,0.20]))
print('Apparent activation Barrier: %.2f eV' % apparent_barrier(473,[0.02,0.20]))


# ## Degree of rate Control

# In[15]:


def degree_of_rate_control(theta0,T,P):
# Prints the solution of the model
    
    diffk_0=0.99   
    diffk_1=1.01 

    XRC=np.zeros(19)
    for i in range(19):
        (kf0,kr0,Ea) = get_rate_constants(T)
        kf0[i]=kf0[i]*diffk_0
        kr0[i]=kr0[i]*diffk_0
        theta = solve_ode(kf0,kr0,theta0,P)
        rates0 = get_rates(theta,kf0,kr0,P)[1]+get_rates(theta,kf0,kr0,P)[6]+get_rates(theta,kf0,kr0,P)[9]+get_rates(theta,kf0,kr0,P)[15]
            
        (kf1,kr1,Ea) = get_rate_constants(T)
        kf1[i]=kf1[i]*diffk_1
        kr1[i]=kr1[i]*diffk_1
        theta = solve_ode(kf1,kr1,theta0,P)
#         rates1 = get_rates(theta,kf1,kr1,P)
        rates1 = get_rates(theta,kf1,kr1,P)[1]+get_rates(theta,kf1,kr1,P)[6]+get_rates(theta,kf1,kr1,P)[9]+get_rates(theta,kf1,kr1,P)[15]
#         print(rates0,rates1)
        
        XRC[i] = (math.log(rates1/rates0))/(math.log(kf1[i]/kf0[i])) 
        print(rates0,rates1,np.round(XRC[i],3))
    return (np.round(XRC,3))


# In[16]:


theta0 = np.zeros(15) # initial guess coverage of CO*, CO2+v*, vac*, O2v*, O*, COO*, CO2* respectively

degree_of_rate_control(theta0,473,[0.02,0.20])

