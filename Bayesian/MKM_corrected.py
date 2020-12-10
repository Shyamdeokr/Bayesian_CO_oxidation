#!/usr/bin/env python
# coding: utf-8

# # Microkinetic Model for CO Oxidation (Pd), (PdO) and (PdO2) + O2 ads - corrected for DFT energetics and barriers

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
from MKM import *


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


# ## Rate constants corrected

# In[4]:


def get_rate_constants_corrected(T,delE,dEa):
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
    dE = np.zeros(19)                    # array initialization
    dE[0] = delE[0]                # CO adsorption (-1.09)
    dE[2] = delE[1]                 # 1st CO2 desorption (0.52)
    dE[3] = delE[2]                 # O2 adsorption in a vacancy(-2.07)
    dE[4] = delE[3]                 # O2 dissociation in the vacancy(-1.29)
    dE[5] = delE[4]                 # 2nd CO adsorption (-0.65)
    dE[6] = delE[5]                 # 2nd CO oxidation(-1.46)
    dE[7] = delE[6]               # 2nd CO2 desorption(-0.0057)
    dE[1] = -6.49-(dE[0]+dE[2]+dE[3]+dE[4]+dE[5]+dE[6]+dE[7])  # 1st CO oxidation (-0.22)
    
    dE[8] = delE[4]         # rxn 6'- 2nd CO adsorption using lattice O
    dE[10] = delE[7]                 # rxn 8' - 2nd CO2 desorption leading to (V+O)*
    dE[11] = delE[8]                # rxn 9 - O2 adsorption at (v+O)*
    dE[12] = delE[9]                # rxn 10 -O2 dissociation at (v+O)*
    dE[13] = delE[10]               # rxn 11 - O migration from (V+O)* --> *
    dE[14] = delE[11]               # rxn 12 - CO adsorption at 2O*
    dE[15] = delE[12]                # rxn 13 - CO oxidation at 2O*
    dE[16] = delE[13]                # rxn 14 - CO2 desorption from O*
    dE[9] =  -6.49-(dE[8]+dE[10]+dE[11]+dE[12]+dE[13]+dE[14]+dE[15]+dE[16])   # rxn 7' - 2nd CO oxidation using lattice O

    dE[17] = delE[14]              # rxn 15 - O2 adsorption on *
    dE[18] = delE[15]                # rxn 16 - O2 dissociation on * to give 2O*



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
    Ea[1] =  dEa[1]               # 1st CO Oxidation barrier = 0.49
    Ea[4] =  dEa[4]                 # O2 dissociation barrier at (v+O)*
    Ea[6] =  dEa[6]                 # 2nd CO Oxidation barrier (assumed)
    Ea[9] =  dEa[9]                 # 2nd CO Oxidation barrier using lattice O
    Ea[12] = dEa[12]                 # O2 dissociation barrier at (v+O)*
    Ea[13] = dEa[13]                 # O migration barrier from (v+O)* ---> *
    Ea[15] = dEa[15]                  # rxn 13 - CO oxidation at 2O*
    Ea[18] = dEa[18]                 # rxn 16 - O2 dissociation on * to give 2O*
    
    

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


# In[5]:


# theta0 = (0.0, 0., 0., 0 , 0 , 0 , 0.0) # initial coverage of CO*, CO2+v*, vac*, O2v*, O*, COO*, CO2*, COO(L)*,CO2(L)*, (v+O)*,(O2+v+O)*, 2O*, (CO+2O)*, (CO2+O)*, O2* respectively
theta0 =np.zeros(15)


# ## Corrected rates and intermediate coverage

# And we call the function with our output values, i.e., the last row of the result matrix $\theta$.

# In[6]:


def print_output_corrected(theta0,T,P,delE,dEa):
# Prints the solution of the model
    (kf,kr,Ea) = get_rate_constants_corrected(T,delE,dEa)
    theta = solve_ode(kf,kr,theta0,P)
    rates   = get_rates(theta,kf,kr,P)
#     print ("the result is:")
#     print
#     for r,rate in enumerate(rates):
#         if r in [0,3,5]:
#             print ("Step",r,": rate =",rate,", kf =",kf[r],", kr=",kr[r],", reverse Ea =",Ea[r])
#         else:
#             print("Step",r,": rate =",rate,", kf =",kf[r],", kr=",kr[r],", Ea =",Ea[r])
#     print ("The coverages for CO*, CO2+v*, vac*, O2v*, O*, COO*, CO2* are:")
#     for t in theta:
#         print (t)
    return (rates[1]+rates[6]+rates[9]+rates[15],theta[0])


# ## Corrected Reaction orders ($P_{CO}$ & $P_{O2}$) and apparent barrier

# In[7]:


def rxn_order_CO_corrected(T,P,delE,dEa):
    gridpoints = 3
    rate_PCO=np.zeros(gridpoints)
    PCO1=P[0]
    PCO2=PCO1+0.05
    PCO_range = np.linspace(PCO1,PCO2,gridpoints)
    for i,PCO in enumerate(PCO_range):
        rate_PCO[i]=print_output_corrected(theta0,T,[PCO,P[1]],delE,dEa)[0] # PO2 = 0.20 
        if rate_PCO[i]<10**-323:
            rate_PCO[i]=10**-323
    PCO_range = PCO_range.reshape(-1, 1)
    LR_CO=model.fit(np.log(PCO_range), np.log(rate_PCO))
    order_CO=LR_CO.coef_    # LR.intercept_
    return order_CO[0]


# In[8]:


def rxn_order_O2_corrected(T,P,delE,dEa):
    gridpoints = 3
    rate_PO2=np.zeros(gridpoints)
    PO21=P[1]
    PO22=PO21+0.1
    PO2_range = np.linspace(PO21,PO22,gridpoints)
    for i,PO2 in enumerate(PO2_range):
        rate_PO2[i]=print_output_corrected(theta0,T,[P[0],PO2],delE,dEa)[0] #PCO = 0.02
        if rate_PO2[i]<10**-323:
            rate_PO2[i]=10**-323
    PO2_range = PO2_range.reshape(-1, 1)
    LR_O2=model.fit(np.log(PO2_range), np.log(rate_PO2))
    order_O2=LR_O2.coef_ # LR.intercept_
    return order_O2[0]


# ## Corrected Apparent barrier

# In[9]:


def apparent_barrier_corrected(T,P,delE,dEa):
    gridpoints = 3
    rate_app=np.zeros(gridpoints)
    T1=T-1
    T2=T+1
    T_range = np.linspace(T1,T2,gridpoints)
    for i,T in enumerate(T_range):
        rate_app[i]=print_output_corrected(theta0,T,P,delE,dEa)[0] # PCO and PO2 are 0.02 and 0.20 respectively
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


# ## Corrected Degree of rate control

# In[10]:


def degree_of_rate_control_corrected(theta0,T,P,delE,dEa):
# Prints the solution of the model
    
    diffk_0=0.99   
    diffk_1=1.01 

    XRC=np.zeros(19)
    for i in range(19):
        (kf0,kr0,Ea) = get_rate_constants_corrected(T,delE,dEa)
        kf0[i]=kf0[i]*diffk_0
        kr0[i]=kr0[i]*diffk_0
        theta = solve_ode(kf0,kr0,theta0,P)
        rates0 = get_rates(theta,kf0,kr0,P)[1]+get_rates(theta,kf0,kr0,P)[6]+get_rates(theta,kf0,kr0,P)[9]+get_rates(theta,kf0,kr0,P)[15]
            
        (kf1,kr1,Ea) = get_rate_constants_corrected(T,delE,dEa)
        kf1[i]=kf1[i]*diffk_1
        kr1[i]=kr1[i]*diffk_1
        theta = solve_ode(kf1,kr1,theta0,P)
#         rates1 = get_rates(theta,kf1,kr1,P)
        rates1 = get_rates(theta,kf1,kr1,P)[1]+get_rates(theta,kf1,kr1,P)[6]+get_rates(theta,kf1,kr1,P)[9]+get_rates(theta,kf1,kr1,P)[15]
#         print(rates0,rates1)
        
        XRC[i] = (math.log(rates1/rates0))/(math.log(kf1[i]/kf0[i])) 
#         print("step", i+1, " ",rates0," ", rates1," ",np.round(XRC[i],3))
    return (XRC)


# In[ ]:




