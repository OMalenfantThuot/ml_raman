# ml_raman
This repository provides the workflow for calculating Raman spectra using machine learning models. The methodology was introduced in the following paper: **[Add Paper Title and Link Here]**.
## Contents 
1. Introduction
2. Installation
3. Examples
    - Graphene
    - Hexagonal Boron Nitride (hBN)
        
## Introduction
Raman spectroscopy is a non-destructive technique used to analyze the vibrational properties of materials. This package implements the workflow described in **[Add Paper Title and Link Here]** , which combines machine-learned interatomic potentials with the Raman-active Γ-weighted density of states method, the latter originally introduced by 
[Hashemi _et al._](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.3.023806). The workflow also incorporates a novel patching technique that allows simulations to scale to systems with tens of thousands of atoms. This repository includes all necessary scripts to reproduce the results for graphene and hexagonal boron nitride in `scripts/` directory. It also contains data and scripts to reproduce figures, located in `figures/` directory. For more detailed information on the theoretical framework and methodology, refer to sections II and III in **[Add Paper Title and Link Here]**.
## Installation 
To install the necessary packages and dependencies, follow these steps:
1. Install [ML_Calc_Driver](https://github.com/OMalenfantThuot/ML_Calc_Driver) and SchnetPack from [this fork](https://github.com/dounia-shaaban/schnetpack)
2. Clone the repository to local machine.
3. `pip install -e .`
**Optional Dependencies:**
-   [`scripts_raman`](https://github.com/OMalenfantThuot/scripts_raman.git): Required for model training and automatic grid selection.
## Examples
We provide examples for calculating Raman spectra for:
-   [**Graphene**](#graphene): G peak calculation in $\mathrm{^{13}C}$ isotope-enriched graphene.
-   [**Hexagonal Boron Nitride (hBN)**](#hbn): Raman intensities calculations in hBN with boron vacancies.

Each script can be run with the `--help` option to display a description of all available input parameters.
It is assumed that the user has a trained machine learning model, which can be obtained using scripts from [scripts_raman](https://github.com/OMalenfantThuot/scripts_raman.git) repository.
### Graphene 
#### Generating supercells
To generate a $100\times100$ graphene supercell with 25% concentration of $\text{C}_{13}$ , run:
   

     python ml_raman/scripts/generate_isotope_sc.py graphene_primitive_cell.cif 100 100 1 0.25 C12 C13 100_100_0p25.mcl

The script assumes that you have a `graphene_primitive_cell.cif` file. The output is in Posinp format (see [mlcalcdriver](https://github.com/OMalenfantThuot/ML_Calc_Driver.git) for details on format).
#### Computing the Hessian matrix
To compute Hessian matrix with the patches method, run

    python ml_raman/scripts/phonon_isotopes_grid.py hessian 100_100_0p25.mcl best_model --device cuda --hessian_savepath hessian_100_100 --grid  5 --sparse

In this example, a $5\times 5$ grid is chosen, and a pre-trained model named `best_model` is used. If no specific grid is given, the grid will be chosen by the code. This, however, requires installing [scripts_raman](https://github.com/OMalenfantThuot/scripts_raman.git).
#### Computing phonon modes
To compute the phonon modes, run:
   

     python -u scripts/lanczos_projections.py 100_100_0p25.mcl best_model graphene --device cuda --results_savepath phonons_100_100  --grid  5

#### Computing Raman intensities
This command computes the Raman spectrum to which is applied a Lorentzian smearing ($\gamma=3$), and the peak Lorentzian fit. It also generates a plot of the Raman intensities:
   

     python scripts/raman/projections_vac.py C 10000  100_100_0p25.mcl phonons_100_100.h5 3 raman_100_100  --smearing Lorentz --save_fit  --save_raman  --npts  2000

  
### hBN
We provide an example of Raman intensities calculations in $\text{hBN}$ with $\text{B}$-type vacancies. 
#### Generating vacancy indicies file
To generate the indices file for a 100x100 supercell with randomly distributed $\text{B}$-type vacancies with a $5\%$ concentration, run the command
  
    python 10000 0.05 B --distribution random --save_name Bvacs_0p05_rand
   
#### Generating supercells
To generate a 100x100 supercell with randomly distributed $\text{B}$-type vacancies corresponding to indices previously generated, run the command
 
    python ml_raman/scripts/generate_vac_sc.py hBN.cif 100 100  B_100_100_0p05  --relax  --model  best_model --device cuda --output_format mlc --vacancies_file  Bvacs_0p05_rand.npy  --max_iter  5000  --step  0.008
The script assumes that you have a `hBN.cif` file. Note that the structure is relaxed using model after vacancies are created.
#### Computing the Hessian matrix
    python ml_raman/scripts/phonon_isotopes_grid.py hessian  B_100_100_0p05.mlc best_model --device cuda --hessian_savepath hessian_B_100_100_0p05 --grid  5 --sparse
#### Computing phonon modes
    python ml_raman/scripts/phonon_isotopes_grid.py phonon B_100_100_0p05.mlc best_model --hessian_path  hessian_B_100_100_0p05.npz --results_savepath phonons_B_100_100_0p05
#### Computing Raman intensities
    python scripts/raman/projections_vac.py BN 100  B_100_100_0p05.mlc phonons_B_100_100_0p05.h5 3 raman_B_100_100_0p05 --smearing Lorentz --save_fit  --save_raman  --npts  2000  --vacancies_file  Bvacs_0p05_rand.npy
