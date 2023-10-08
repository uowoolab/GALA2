
### GALAV2
![UOttawaLogo](https://github.com/uowoolab/GALA2/blob/main/Images/UOttawaLogo.png) 
---
#### Guest Atom Localization Algorithm (Work in progress)

GALA is a Python program engineered for pinpointing guest atoms at the binding sites of a framework, utilizing GCMC simulated probability plots.

#### Current Compatibility
GALA currently supports FastMC  GCMC simulation probability plots. Integration with RASPA is under development. 

#### Evolution
Initially, GALA was embedded within FAPS and depended on proprietary functions/methods, supporting a limited number of guests. It has now evolved into an independent code, boasting compatibility with an extensive range of guests, contingent upon appropriate specification in the requisite input files.

---

### Features
- **Extensive Guest Compatibility:** Works with any guest, given that it’s correctly specified in the input files.
- **Framework Binding Sites Identification:** Utilizes GCMC simulated probability plots for precise guest atom placement within porous materials.
- **Library Integration:** Incorporates well-known libraries like Pymatgen for enhanced functionality and reliability.
- **Flexible Execution:** Can be run manually for individual directories or in a high-throughput manner for bulk processing, offering scalability.
- **Open-Source:** The code is freely available for modification and distribution, promoting collaboration and innovation.
- **User-Friendly:** Accommodates both novice and expert users with an simple singular GALA.inp input file.
- **Customizable:** Allows users to easily adjust parameters to fit specific requirements and conditions.
- **Community Support:** Being open-source, it encourages contributions from the community for continuous improvement and updates.

### Dependencies
Ensure you have the following packages installed. The versions specified below are recommended to guarantee compatibility. *TESTED*

Python - 3.9.7
numpy - 1.20.3
pymatgen - 2023.8.10
scipy - 1.7.1
You can install these packages using pip:

```bash
pip install numpy==1.20.3 pymatgen==2023.8.10 scipy==1.7.1
```

### Imported Modules and Packages
GALA imports the following modules and packages to execute its functions efficiently. The listed ones are either built-in or installed from the dependencies above.

Built-in:
```bash
os, itertools, shutil, subprocess, multiprocessing, time
```

Installed:
```bash
numpy as np
pymatgen.core (Element, Molecule, Structure)
pymatgen.io.common (VolumetricData)
pymatgen.util.coord (all_distances, pbc_shortest_vectors)
pymatgen.analysis.structure_matcher (StructureMatcher)
pymatgen.io.cif (CifWriter)
scipy.ndimage (gaussian_filter, maximum_filter, generate_binary_structure, binary_erosion, iterate_structure)
```

### Installation
```bash
git clone https://github.com/uowoolab/GALA2.git
```

### Usage

1. Prepare the input files as per the specifications using your prefered text editor.
```bash
nano GALA.inp; vim GALA.inp; ...
```
2. Run the program using the command:
```bash
python GALA_main.py
```

### Input Files
Required input files:
```bash
GALA.inp
FIELD
Prob_Guest_X_Site_Y.cube or Prob_Guest_X_Site_Y_folded.cube
```
### Input File Configuration

Ensure to configure the input file accurately to execute GALA effectively. Below are the parameters and options available:

#### General Input

- **Simulation Framework Selection:** (Required)
  - Choose between FASTMC or RASPA (RASPA is not supported in this version).
  - Example: `FASTMC` 

- **Input Files Directory Path:** (Default: Current Working Directory (CWD))
  - Specify the absolute path containing the required FIELD and probability plots.
  - Options: `/your/absolute/path`, `/leave/empty`

- **Binding Site Cap:** (Default: Infinity (all binding sites))
  - Define the number of binding sites used for energy calculations.
  - Example: `5` (non-zero positive integer)

#### MD Parameters

Certainly! Here’s the revised MD Parameters section with the added information about the dedicated CPU for parallelization.

#### MD Parameters

- **MD Program Executable:** (Required)
  - Provide the location or name of the executable for running Molecular Dynamics simulations.
  - Example: `/path/to/executable`

- **Cutoff (in Angstroms):** (Requited)
  - Set the distance cutoff for long-ranged interactions in the simulation.
  - Example: `12.0`

- **Dedicated CPU for Parallelization of MD Jobs:** (Default: `1`)
  - Specify the number of CPUs dedicated to parallelizing Molecular Dynamics jobs.
  - Example: `1`

#### GCMC Analysis Parameters

- **Sigma:** (Required)
  - State the standard deviation for the Gaussian filter during data analysis.
  - Example: `2`

- **Radius:** (Required)
  - Specify the footprint radius used during filtering.
  - Example: `0.45`

- **Cutoff:** (Required)
  - Determine the threshold below which probability distribution peaks are ignored.
  - Example: `0.1`

- **Write Folded (T/F):** (Default: `F`)
  - Decide if the folded probability plots will be generated.
  - Options: `T`, `F`

- **Grid Factors:** (Default: `1 1 1`)
  - Provide the factors to convert the supercell probability plot to its unit cell (required if "Write Folded" is `True` or unfolded plots are provided).
  - Example: `4 3 3`

- **Write Smoothed (T/F):** (Default: `F`)
  - Indicate if smoothed probability plots should be generated.
  - Options: `T`, `F`

#### Probability Plot Selection

- **Guest Molecule Formula:** (Required)
  - List the formulas of guest molecules of interest.
  - Example: `CO2 N2 H2O`

- **Sites:** (Required)
  - Enumerate specific sites associated with each guest molecule, grouped accordingly.
  - Example: `Cx,Ox COM,Nx Hx,Ox,L`

#### Post Process Cleaning

- **DL_POLY_BS Directory:** (Default: `F`)
  - Instructions on removing the directory containing DL_POLY individual jobs to clean up post-processing (removed DL_POLY_BS).
  - Options: `T`, `F`

### Output
#### Binding Sites in Framework
- **File:** `X_binding_site.cif`
- **Description:** This CIF file offers a visual representation of the highest occupancy guest binding sites within the framework. It portrays the unit cell integrated with binding sites calculated by the GALA code.

#### Binding Site + Energies
- **File:** `X_gala_binding_sites.xyz`
- **Description:** This XYZ file delivers detailed information from the DL Poly calculations. It encompasses the binding energy, electrostatic potential percentage, Vander Waals energy, electrostatic potential energy, and the relative occupancy (compared to the highest occupancy) for each binding site.

#### Temporary Files
- **DL_POLY.output:**
  - **Description:** A temporary file that logs error codes if a DL Poly calculation fails. If the calculation is successful, a “DL Poly terminated successfully” message is printed in the terminal, and the file is deleted. For failed jobs, it provides the last four lines of the output to aid in diagnosing the failure.

- **GALA.log:** 
  - **Description:** Details to be added soon.

### License
Place holder, do we have a liscence for this?
