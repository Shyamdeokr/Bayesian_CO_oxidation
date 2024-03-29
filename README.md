# Microkinetic Model (MKM)
> This python code simulates the Pd site dynamics on CeO<sub>2</sub>(100) under lean CO oxidation, TOF for CO oxidation, Reaction orders in CO and O<sub>2</sub> as well as the rate limiting step and the coverage of the intermediate species under reaction conditions of the following extensive reaction network.

> Used in the following paper - https://pubs.acs.org/doi/abs/10.1021/acscatal.2c03194 
> (Emergent Behavior in Oxidation Catalysis over Single-Atom Pd on a Reducible CeO2 Support via Mixed Redox Cycles) to generate corrections in DFT calculated energetics and activation barriers

> Further details on the MKM and the Bayesian Inference can be found in the Supplementary Information of the paper.

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

delE and dEa which are corrections to input DFT reaction energies and DFT barriers respectively, can be changed in Bayesian.py. The code can be modified to input DFT energetics from the user.

It outputs Bayesian correted energetics (MCMC iterations and distribution plots) that match experimental data in an output folder.
