## MKM.py - original MKM model for the given reaction network

## MKM-corrected.py - Corrections to the energetics for the elementary steps of the reaction network

## Bayesian.py - Bayesian corrected DFT energetics (interacts with MKM and MKM-corrected). Needs DFT energetics, T, P and experimental reaction orders in CO and O2 as inputs and outputs Bayesian correted energetics that match experimental data

## Usage -

> python Bayesian.py 

> delE, dEa and number of MCMC iterations which are corrections to input DFT reaction energies and DFT barriers respectively, can be changed in Bayesian.py. The code can be modified to input energetics from user.

It outputs Bayesian correted energetics (MCMC iterations and distribution plots) that match experimental data in an output folder.
