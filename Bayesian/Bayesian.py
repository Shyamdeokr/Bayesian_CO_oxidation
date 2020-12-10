#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from MKM import *
from MKM_corrected import *


# In[2]:


def rxnOrderCO(T,P,delE,dEa):
    rxnOrderCO = rxn_order_CO_corrected(T,P,delE,dEa)
    return rxnOrderCO

def rxnOrderO2(T,P,delE,dEa):
    rxnOrderO2 = rxn_order_O2_corrected(T,P,delE,dEa)
    return rxnOrderO2

def act_barrier(T,P,delE,dEa):
    act_barrier = apparent_barrier_corrected(T,P,delE,dEa)
    return act_barrier


# ## Markov chain MCMC algorithm (Prior, Likelihood and posterior)

# In[9]:


def MCMC(iterations,delE, dEa):
    #delE=array of coorection(16 elements) - initial values
    # and dEa(19 elements)
    # Ea (1,4,6,9,12,13,15,18) - 1st CO, O2 dissociation and 2nd CO barrier - initial values

    delE_old,delE_PDF_old,delE_new,delE_PDF_new=np.zeros((4,16))
    dEa_old,dEa_PDF_old,dEa_new,dEa_PDF_new=np.zeros((4,19))
    delE_record,dEa_record=[],[]
    delE_pos_new,delE_pos_old=np.zeros((2,16))
    dEa_pos_new,dEa_pos_old=np.zeros((2,19))
    
    rxnOrderCO_record =[]
    rxnOrderO2_record=[]
    act_barrier_record=[]


    for j in range(iterations):
        for i in range(16):
#             delE_old[i]=delE[i] #change to record
#             delE_PDF_old[i]=scipy.stats.norm(loc=delE_old[i], scale=0.1).pdf(delE_old[i])
            if i in [0,2,4,8,11,14]: # adsorption steps - 
                delE_old[i]=delE[i] #change to record
                mu, sigma = delE_old[i], 0.1 # mu = mean, sigma = standard deviation
                a, b = (-2 - mu) / sigma, (0 - mu) / sigma
                delE_PDF_old[i]=scipy.stats.truncnorm(a,b,loc=delE_old[i],scale=sigma).pdf(delE_old[i])
                delE_new[i]=scipy.stats.truncnorm.rvs(a,b,loc=delE_old[i],scale=sigma)
                assert np.all(delE_new[i] >= -2)
                assert np.all(delE_new[i] <= 0)
                delE_PDF_new[i]=scipy.stats.truncnorm(a,b,loc=delE_old[i],scale=sigma).pdf(delE_new[i])
            
            elif i in [1,6,7,13]: # desorption steps 
                delE_old[i]=delE[i] #change to record
                mu, sigma = delE_old[i], 0.1 # mu = mean, sigma = standard deviation
                a, b = (0 - mu) / sigma, (2 - mu) / sigma
                delE_PDF_old[i]=scipy.stats.truncnorm(a,b,loc=delE_old[i],scale=sigma).pdf(delE_old[i])
                delE_new[i]=scipy.stats.truncnorm.rvs(a,b,loc=delE_old[i],scale=sigma)
                assert np.all(delE_new[i] >= 0)
                assert np.all(delE_new[i] <= 2)
#                 delE_new[i]=scipy.stats.truncnorm.rvs(a,b,loc=delE_old[i],scale=sigma)
                delE_PDF_new[i]=scipy.stats.truncnorm(a,b,loc=delE_old[i],scale=sigma).pdf(delE_new[i])
           
            else: # surface reactions
#                 print(delE[i])
                delE_old[i]=delE[i] #change to record
                mu, sigma = delE_old[i], 0.1 # mu = mean, sigma = standard deviation
                a, b = (-2 - mu) / sigma, (2 - mu) / sigma
                delE_PDF_old[i]=scipy.stats.truncnorm(a,b,loc=delE_old[i],scale=sigma).pdf(delE_old[i])
                delE_new[i]=scipy.stats.truncnorm.rvs(a,b,loc=delE_old[i],scale=sigma)
                assert np.all(delE_new[i] >= -2)
                assert np.all(delE_new[i] <= 2)
#                 delE_new[i]=scipy.stats.truncnorm.rvs(a,b,loc=delE_old[i],scale=sigma)
                delE_PDF_new[i]=scipy.stats.truncnorm(a,b,loc=delE_old[i],scale=sigma).pdf(delE_new[i])
#                 print(delE_PDF_new[i], delE_PDF_old[i])  
#         print(delE_new)
#         print(delE_old)


        for i in [1,4,6,9,12,13,15,18]:
            dEa_old[i]=dEa[i]
            dEa_PDF_old[i]=scipy.stats.truncnorm(a,b,loc=dEa_old[i],scale=sigma).pdf(dEa_old[i])
            mu, sigma = dEa_old[i], 0.1 # mu = mean, sigma = standard deviation
            a, b = (0 - mu) / sigma, (1 - mu) / sigma
            dEa_new[i]=scipy.stats.truncnorm.rvs(a,b,loc=dEa_old[i],scale=sigma)
            assert np.all(dEa_new[i] >= 0)
            assert np.all(dEa_new[i] <= 1)
            dEa_new[i]=scipy.stats.truncnorm.rvs(a,b,loc=dEa_old[i],scale=sigma)
            dEa_PDF_new[i]=scipy.stats.truncnorm(a,b,loc=dEa_old[i],scale=sigma).pdf(dEa_new[i])

#         print(dEa_new)
#         print(dEa_old)
        
        rxnOrderCO_old=rxnOrderCO(T,P,delE_old,dEa_old)
        rxnOrderO2_old=rxnOrderO2(T,P,delE_old,dEa_old)
        act_barrier_old=act_barrier(T,P,delE_old,dEa_old)

        rxnOrderCO_new=rxnOrderCO(T,P,delE_new,dEa_new)
        rxnOrderO2_new=rxnOrderO2(T,P,delE_new,dEa_new)
        act_barrier_new=act_barrier(T,P,delE_new,dEa_new)
        
#         print(rxnOrderCO_new,rxnOrderCO_old)

        Likelihood_new=scipy.stats.norm(loc=rxnOrderCO_new, scale=0.05).pdf(exp_rxn_order_CO)*scipy.stats.norm(loc=rxnOrderO2_new, scale=0.05).pdf(exp_rxn_order_O2)*scipy.stats.norm(loc=act_barrier_new, scale=0.05).pdf(exp_act_barrier)
        Likelihood_old=scipy.stats.norm(loc=rxnOrderCO_old, scale=0.05).pdf(exp_rxn_order_CO)*scipy.stats.norm(loc=rxnOrderO2_old, scale=0.05).pdf(exp_rxn_order_O2)*scipy.stats.norm(loc=act_barrier_old, scale=0.05).pdf(exp_act_barrier)

#         print(Likelihood_new, Likelihood_old)
        
        for i in range(len(delE)):
            delE_pos_new[i]=Likelihood_new*delE_PDF_new[i]
            delE_pos_old[i]=Likelihood_old*delE_PDF_old[i]

        for i in [1,4,6,9,12,13,15,18]:
            dEa_pos_new[i]=Likelihood_new*dEa_PDF_new[i]
            dEa_pos_old[i]=Likelihood_old*dEa_PDF_old[i]

        # delE_old,delE_new
        # dEa_old,dEa_new
        

        delE_record_tmp,dEa_record_tmp=[],[]

        for i in range(len(delE)):
            if delE_pos_new[i]/delE_pos_old[i]>np.random.random():
                delE[i]=delE_new[i]  
            else:
                delE[i]=delE_old[i]
#             print(i,delE_pos_new[i],delE_pos_old[i])
            delE_record_tmp.append(delE[i])
        delE_record.append(delE_record_tmp)

        for i in range(len(dEa)):
            if i in [1,4,6,9,12,13,15,18]:
                if dEa_pos_new[i]/dEa_pos_old[i]>np.random.random():
                    dEa[i]=dEa_new[i]
                else:
                    dEa[i]=dEa_old[i]
#                 print(dEa_pos_new[i],dEa_pos_old[i])
            else:
                dEa[i]=0
            dEa_record_tmp.append(dEa[i])
        dEa_record.append(dEa_record_tmp)

        rxnOrderCO_record.append(rxnOrderCO(T,P,delE,dEa))
        rxnOrderO2_record.append(rxnOrderO2(T,P,delE,dEa))
        act_barrier_record.append(act_barrier(T,P,delE,dEa))
        
        print(f"{j}", end =" ") 


    prior=[delE_record,dEa_record]
    posterior=[rxnOrderCO_record, rxnOrderO2_record, act_barrier_record]
        
        
    return prior,posterior


# ## Experimental conditions and observables

# In[4]:


T = 473
P = [0.02, 0.2] 

exp_rxn_order_CO = 1.3
exp_rxn_order_O2 = -0.13
exp_act_barrier = 0.67 # in eV


# ## Data initialization from DFT for corrections

# In[5]:

## Input/change delE and dEa as DFT energetics to be corrected by the user in the following arrays:

delE=[-1.662,  0.348, -0.167, -1.067, -0.449, -0.64 ,  0.432,  0.726,
        -1.483,  0.438, -1.657, -0.761, -1.667,  0.516, -1.738, -0.3]

dEa=[0.   , 0.336, 0.   , 0.   , 0.488, 0.   , 0.304, 0.   , 0.   ,
        0.365, 0.   , 0.   , 0.685, 0.168, 0.   , 0.881, 0.   , 0.   ,
        0.877]


# In[13]:


Rxn_E=["CO adsorption (-1.09)", "1st CO2 desorption (0.52)","O2 adsorption in a vacancy(-2.07)","O2 dissociation in the vacancy(-1.29)",
      "2nd CO adsorption (-0.65)","2nd CO oxidation(-1.46)","2nd CO2 desorption(-0.0057)","2nd CO2 desorption leading to (V+O)*(0.58)",
      "O2 adsorption at (v+O)*(-2.34)",
      "O2 dissociation at (v+O)*(-0.97)",
      "O migration from (V+O)* --> *(-1.27)",
      "CO adsorption at 2O*(-0.35)",
      "CO oxidation at 2O*(-2.31)",
      "CO2 desorption from O*(0.45)",
      "O2 adsorption on *(-0.0067)",
      "O2 dissociation on * to give 2O*(-1.77)"]
Eact=["1st CO Oxidation barrier(0.49)","2nd CO Oxidation barrier(0.19)","O2 dissociation barrier at (v+O)*_TS_running(0)",
     "2nd CO Oxidation barrier using lattice O(0.69)","O2 dissociation barrier at (v+O)*(0.51)",
     "O migration barrier from (v+O)* ---> *(0.18)",
     "CO oxidation at 2O*(no Eact)",
     "O2 dissociation on * to give 2O*(1.44)"]


# In[31]:


print("The DFT reaction energeies are for following elementary steps:","\n",Rxn_E, end ="\n\n")

print("The DFT activation barriers are for following elementary steps:","\n", Rxn_E, end ="\n\n")

print("Please input in the form of arrays")


# In[6]:


#delE=array of coorection(16 elements) - initial values
# delE=input("Enter delE (correction in DFT reaction energies):")


# In[7]:


# dEa= array of coorection(19 elements) - initial values
# dEa=input("Enter dEa(correction in DFT activation barrier):")


# ## Initiation of MCMC chains

# In[11]:

##Users should input the number of MCMC iterations desired for running bayesian inference in the following line.
iterations=10 # Number of MCMC iterations
sample=MCMC(iterations,delE,dEa)


# In[12]:


# prior=[delE_record,dEa_record]

# posterior=[rxnOrderCO_record, rxnOrderO2_record, act_barrier_record]
prior=[np.round(sample[0][0],3),np.round(sample[0][1],3)]
posterior=[np.round(sample[1][0],3),np.round(sample[1][1],3),np.round(sample[1][2],3)]
prior[0][-1], prior[1][-1]
# posterior[0]


# ## Storing and loading data (pickle)

# In[77]:


import os
if not os.path.exists('output'):
    os.makedirs('output')


# In[78]:


import pickle
filename="./output/MCMC_DFT_energetics_corrected_bayesian"
outfile = open(filename,'wb')
pickle.dump(sample,outfile)
outfile.close()


# In[37]:


import pickle
filename="./output/MCMC_DFT_energetics_corrected_bayesian"
infile = open(filename,'rb')
sample = pickle.load(infile)
infile.close()


# ### Distribution of DFT reaction energies

# In[69]:


# sample=MCMC(10000,np.zeros(16),np.zeros(19))

plt.rcParams["figure.figsize"] = (16,16)
for i in range(len(delE)):
    plt.subplot(len(delE)/4,len(delE)/4,i+1)
    plt.title(Rxn_E[i])
    plt.plot(np.array(prior[0][:,i]), 'g.')
    plt.tight_layout(pad=0.5,h_pad=None)
#    plt.show
    plt.savefig("./output/DFT_reaction_energies_MCMC_posteriors.png")


# In[36]:


plt.rcParams["figure.figsize"] = (16,16)
for i in range(len(delE)):
    plt.subplot(len(delE)/4,len(delE)/4,i+1)
    plt.title(Rxn_E[i])
    plt.hist(np.array(prior[0][:,i]),density=True)
    plt.tight_layout(pad=0.5,h_pad=None)
#    plt.show
    plt.savefig("./output/DFT_reaction_energies_MCMC_posteriors_distribution.png")


# ### Distribution of DFT activation barriers

# In[37]:


# sample=MCMC(10000,np.zeros(16),np.zeros(19))

plt.rcParams["figure.figsize"] = (16,16)
for i,j in enumerate([1,4,6,9,12,13,15,18]):
    plt.subplot(4,2,i+1)
    plt.title(Eact[i])
    plt.plot(np.array(prior[1][:,j]), 'm.')
#    plt.show
    plt.savefig("./output/DFT_activation_barrier_MCMC_posteriors.png")


# In[38]:


plt.rcParams["figure.figsize"] = (16,16)
for i,j in enumerate([1,4,6,9,12,13,15,18]):
    plt.subplot(4,2,i+1)
    plt.title(Eact[i])
    plt.hist(np.array(prior[1][:,j]),density=True)
#    plt.show
    plt.savefig("./output/DFT_activation_barriers_MCMC_posteriors_distribution.png")


# ### Distribution of Reaction orders and barrier

# In[58]:


# Reaction order in CO
# plt.rcParams["figure.figsize"] = (4,4)
plt.rcParams["figure.figsize"] = (16,16)
plt.subplot(3,2,1)
plt.title("Reaction order in CO",fontsize=20)
plt.plot(np.array(posterior[0][:]), 'r.')
plt.xlabel("MCMC iterations")
plt.subplot(3,2,2)
plt.hist(posterior[0][:])
# Reaction order in O2
plt.subplot(3,2,3)
plt.title("Reaction order in O2",fontsize=20)
plt.plot(np.array(posterior[1][:]), 'r.')
plt.xlabel("MCMC iterations")
plt.subplot(3,2,4)
plt.hist(posterior[1][:])
# Apparent barrier
plt.subplot(3,2,5)
plt.title("Apparent barrier",fontsize=20)
plt.plot(np.array(posterior[2][:]), 'r.')
plt.xlabel("MCMC iterations")
plt.subplot(3,2,6)
plt.hist(posterior[2][:])
plt.tight_layout(pad=2,h_pad=None)
plt.savefig("./output/rxnorders&barriers_Bayesian.png")


# In[ ]:




