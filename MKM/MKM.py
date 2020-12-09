#!/usr/bin/env python
# coding: utf-8

# In[1]:

import math
import scipy.stats
import numpy as np
from sklearn.linear_model import LinearRegression
model = LinearRegression()

PCO2 = 0        # bar
R = 8.314
mAr = 39.848
mCO = 28.0101
mO2 = 32
mCO2 = 44

SCOn = 197.660
SO2n = 205.152
SCO2n = 213.79


J2eV = 6.24150974E18            # eV/J
Na   = 6.0221415E23             # mol-1
h    = 6.626068E-34 * J2eV      # in eV*s
kb   = 1.3806503E-23 * J2eV     # in eV/K
# kbT  = kb * T                   # in eV

# theta0 = (0.0, 0., 0., 0 , 0 , 0 , 0.0) # initial coverage of CO*, CO2+v*, vac*, O2v*, O*, COO*, CO2*, COO(L)*,CO2(L)*, (v+O)*,(O2+v+O)*, 2O*, (CO+2O)*, (CO2+O)*, O2* respectively
theta0 =np.zeros(15)

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
    dE[8] = delE[7] # O migration from Pd-far to near to form Pd and regenerate the cycle
    dE[1] = -6.49-(dE[0]+dE[2]+dE[3]+dE[4]+dE[5]+dE[6]+dE[7]+dE[8])   # 1st CO oxidation (-0.22)
#     delE[1] = dE[1]+0.22
    
#     dE[8] = delE[4]         # rxn 6'- 2nd CO adsorption using lattice O
    dE[10] = delE[8]                 # rxn 8' - 2nd CO2 desorption leading to (V+O)*
    dE[11] = delE[9]                # rxn 9 - O2 adsorption at (v+O)*
    dE[12] = delE[10]                # rxn 10 -O2 dissociation at (v+O)*
    dE[13] = delE[11]               # rxn 11 - O migration from (V+O)* --> *
    dE[14] = delE[12]               # rxn 12 - CO adsorption at 2O*
    dE[15] = delE[13]                # rxn 13 - CO oxidation at 2O*
    dE[16] = delE[14]                # rxn 14 - CO2 desorption from O*
    dE[9] =  -6.49-(dE[5]+dE[10]+dE[11]+dE[12]+dE[13]+dE[14]+dE[15]+dE[16])   # rxn 7' - 2nd CO oxidation using lattice O

    dE[17] = delE[15]              # rxn 15 - O2 adsorption on *
    dE[18] = delE[16]                # rxn 16 - O2 dissociation on * to give 2O*

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
    dS[8] = 0
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
    Ea[1] =  dEa[1]                 # 1st CO Oxidation barrier = 0.49
    Ea[4] =  dEa[4]                     # O2 dissociation barrier
    Ea[6] =  dEa[6]                 # 2nd CO Oxidation barrier (using far-off O)
    Ea[8] =  dEa[8]      # O migration from Pd-far to near to form Pd and regenerate the cycle
                        # from Hansens paper (TS still running)
    Ea[9] = dEa[9]                # 2nd CO Oxidation barrier using lattice O (0.69)
    Ea[12] = dEa[12]                 # O2 dissociation barrier at (v+O)*
       
    Ea[13] = dEa[13]               # O migration barrier from (v+O)* ---> *
    Ea[15] = dEa[15]                    # CO oxidation at 2O*
    Ea[18] = dEa[18]                 # O2 dissociation on * to give 2O*

    # Entropy changes to the transition state
    STS = np.zeros(19)             # array initialization                  
    STS[0] =  (0.30*SCOn/R + 3.3-1/3*(18.6+math.log((mCO/mAr)**1.5*(T/298)**2.5)))*R      # 1st CO adsorption entropy - ignoring 
    # SCOn dependence on T
    STS[2] =  (0.30*SCO2n/R + 3.3-1/3*(18.6+math.log((mCO2/mAr)**1.5*(T/298)**2.5)))*kb      # 1st CO2 adsorption entropy           
    STS[3] =  (0.30*SO2n/R + 3.3-1/3*(18.6+math.log((mO2/mAr)**1.5*(T/298)**2.5)))*R      # O2 adsorption entropy    
    STS[5] =  STS[0]
       
    STS[14]=  STS[0]      # CO adsorption entropies,
    STS[7] =  STS[2]      # CO2 adsorption entropies
    STS[10]= STS[16] = STS[2]   # CO2 adsorption entropies
    STS[17]= STS[11] = STS[3]   # O2 adsorption entropy    

    # Calculate equilibrium and rate constants
    K  = [0]*19                   # equilibrium constants
    kf = [0]*19                   # forward rate constants
    kr = [0]*19                   # reverse rate constants
    for i in range(19):
        dG = dE[i] - T*dS[i]
        K[i]  = np.exp(-dG/kbT)
        
        # Enforce Ea > 0, and Ea > dE 
        if i not in [0,3,5,11,14,17]: #(steps are adsorption/desorption steps)
            Ea[i] = max([0,dE[i],Ea[i]])   
            kf[i] = kbT/h * np.exp(STS[i]/kb) * np.exp(-Ea[i]/kbT)
            kr[i] = kf[i]/K[i] # enforce thermodynamic consistency
        else:
            Ea[i] =-dE[i]                                         # Ea[i] = Eads
            kr[i] = kbT/h * np.exp(STS[i]/R) * np.exp(-Ea[i]/kbT) # STS = TS-ads for adsorption 0,3 and 5 
            kf[i] = K[i]*kr[i]
            
    return (kf,kr,Ea)  


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
    tvacfar = theta[7]

#     tCOOL = theta[7]
    tCO2L = theta[8]
    tvplusO = theta[9]
    tO2plusvplusO = theta[10]
    t2O = theta[11]
    tCOplus2O = theta[12]
    tCO2plusO = theta[13]
    tO2 = theta[14]
    tstar = 1.0 - tCO - tCO2v - tvac - tO2v - tO - tCOO - tCO2 - tvacfar -tCO2L - tvplusO -tO2plusvplusO -t2O -tCOplus2O -tCO2plusO - tO2  # site balance
    
    PCO = P[0]
    PO2 = P[1]

    # Caluclate the rates: 
    rate    = [0]*19               # array with 8 zeros
    rate[0] = kf[0] * PCO * tstar - kr[0] * tCO
    rate[1] = kf[1] * tCO - kr[1] * tCO2v
    rate[2] = kf[2] * tCO2v - kr[2] * PCO2 * tvac
    rate[3] = kf[3] * tvac * PO2 - kr[3] * tO2v
    rate[4] = kf[4] * tO2v - kr[4] * tO
    rate[5] = kf[5] * tO * PCO - kr[5] * tCOO 
    rate[6] = kf[6] * tCOO - kr[6] * tCO2
    rate[7] = kf[7] * tCO2 - kr[7] * PCO2 * tvacfar
    rate[8] = kf[8] * tvacfar - kr[8] * tstar
    
#     rate[8] = kf[8] * tO * PCO - kr[8] * tCOOL
    rate[9] = kf[9] * tCOO - kr[9] * tCO2L
    rate[10] = kf[10] * tCO2L - kr[10] * PCO2 * tvplusO
    
    rate[11] = kf[11] * tvplusO * PO2 - kr[11] * tO2plusvplusO
    rate[12] = kf[12] * tO2plusvplusO - kr[12] * t2O
    rate[13] = kf[13] * tvplusO - kr[13] * tstar
    
    rate[14] = kf[14] * t2O * PCO - kr[14] * tCOplus2O
    rate[15] = kf[15] * tCOplus2O - kr[15] * tCO2plusO
    rate[16] = kf[16] * tCO2plusO - kr[16] * PCO2 * tO
    
    rate[17] = kf[17] * tstar * PO2 - kr[17] * tO2
    rate[18] = kf[18] * tO2 - kr[18] * t2O
    
#     print("rates",rate)
    
    return rate 


def get_odes(theta,t,kf,kr,P):
# returns the system of ODEs d(theta)/dt, calculated at the current value of theta (and time t)

    rate = get_rates(theta,kf,kr,P)       # calculate rates at current value of theta

    # Time derivatives of theta
    dt  = [0]*15
    dt[0] = rate[0] - rate[1]           # d(tCO)/dt
    dt[1] = rate[1] - rate[2]           
    dt[2] = rate[2] - rate[3]  
    dt[3] = rate[3] - rate[4] 
    dt[4] = rate[4] - rate[5] + rate[16]       #d(tO)/dt
    dt[5] = rate[5] - rate[6] - rate[9]                 #d(tCOO)/dt
    dt[6] = rate[6] - rate[7] 
    dt[7] = rate[7] - rate[8]

#     dt[7] = rate[8] - rate[9]
    dt[8] = rate[9] - rate[10]
    
    dt[9] = rate[10] - rate[11] -rate[13]  #d(v+O)/dt
    
    dt[10] = rate[11] - rate[12]
    
    dt[11] = rate[12] - rate[14] + rate[18] #d(t2O)/dt
    
    dt[12] = rate[14] - rate[15]
    dt[13] = rate[15] - rate[16]
    dt[14] = rate[17] - rate[18]

    
    return dt


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

def rxn_order_CO_corrected(T,P,delE,dEa):
    gridpoints = 3
    rate_PCO=np.zeros(gridpoints)
    PCO1=P[0]
    PCO2=PCO1+0.05
    PCO_range = np.linspace(PCO1,PCO2,gridpoints)
    for i,PCO in enumerate(PCO_range):
        rate_PCO[i]=print_output_corrected(theta0,T,[PCO,P[1]],delE,dEa)[0] # PO2 = 0.20
#         print("rate_PCO",rate_PCO[i])
        if rate_PCO[i]<10**-323:
            rate_PCO[i]=10**-323
#         print("rate_PCO",rate_PCO[i])
    PCO_range = PCO_range.reshape(-1, 1)
    LR_CO=model.fit(np.log(PCO_range), np.log(rate_PCO))
    order_CO=LR_CO.coef_    # LR.intercept_
    return order_CO[0]

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
#         print("rate_PO2",rate_PO2[i])
    PO2_range = PO2_range.reshape(-1, 1)
    LR_O2=model.fit(np.log(PO2_range), np.log(rate_PO2))
    order_O2=LR_O2.coef_ # LR.intercept_
    return order_O2[0]

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

10**-322

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

def rxnOrderCO(T,P,delE,dEa):
    rxnOrderCO = rxn_order_CO_corrected(T,P,delE,dEa)
    return rxnOrderCO

def rxnOrderO2(T,P,delE,dEa):
    rxnOrderO2 = rxn_order_O2_corrected(T,P,delE,dEa)
    return rxnOrderO2

def act_barrier(T,P,delE,dEa):
    act_barrier = apparent_barrier_corrected(T,P,delE,dEa)
    return act_barrier


