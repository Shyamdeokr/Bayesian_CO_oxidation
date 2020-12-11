# Microkinetic Model
> This python code simulates the Pd site dynamics on CeO<sub>2</sub>(100) under lean CO oxidation, TOF for CO oxidation, Reaction orders in CO and O<sub>2</sub> as well as the rate limiting step and the coverage of the intermediate species under reaction conditions of the following extensive reaction network.

## Reaction network for CO Oxidation on Pd sites - (Pd), (PdO) and (PdO<sub>2</sub>) + O<sub>2</sub> adsorption

## Reactions - 1 - 8
# ![plot](./Figures/image1.png)


## Reactions 6' - 11
# ![plot](./Figures/image2.png)

## Reactions 12-14
# ![plot](./Figures/image3.png)

## Reactions 15-16 for O2 adsorption on Pd
# ![plot](./Figures/image4.png)


## Usage -
### Run as following:
python Bayesian.py

delE and dEa which are corrections to input DFT reaction energies and DFT barriers respectively, can be changed in Bayesian.py. code can be modified to input energetics from user.

It outputs Bayesian correted energetics (MCMC iterations and distribution plots) that match experimental data in an output folder.
