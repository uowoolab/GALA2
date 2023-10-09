## How to Run GALA

GALA can be executed manually for small-scale projects or using sbatch for high-throughput, larger scale analyses. Follow the steps below according to your specific needs.

---

### 1. Small Scale (Manual Execution)

#### Preparation
Ensure that the `GALA.inp` and `GALA_main.py` files are located in the same directory. A well-organized directory structure facilitates an efficient execution process.

<p align="center">
  <img src="https://github.com/uowoolab/GALA2/blob/main/Images/Small_Scalle_light.svg" alt="Small_Scale_Workflow" width="500"/>
</p>

#### Execution Steps
1. Open the `GALA.inp` file.
2. Find the comment line `>> Absolute path to directory with input files (FIELD + Probability plots) <<`.
3. Below this comment, insert the absolute path of the directory where you plan to run `GALA_main.py`.
   - Example: To execute GALA in the `MOF1` directory, update the path to the absolute path where `MOF1` is located.
4. Save and close the `GALA.inp` file.
5. In the terminal, navigate to the directory containing both `GALA.inp` and `GALA_main.py`.
6. Execute the command:
   ```bash
   python GALA_main.py
   ```
7. GALA will process using the specified directory settings.

---

### 2. High Throughput (Using sbatch)

#### Preparation
For extensive analysis, utilize sbatch to enable parallel execution and optimize computation time. In this method, leave the absolute path in the `GALA.inp` file empty.

<p align="center">
  <img src="https://github.com/uowoolab/GALA2/blob/main/Images/Large_Scale_light.svg" alt="Large_Scale_Workflow" width="500"/>
</p>

#### Execution Steps
1. Ensure the `GALA_main.py` script is accessible.
2. In the terminal, change to the directory intended for processing.
   - Example: To process the `MOF1` directory, point your terminal to the `MOF1` directory.
3. Execute the command, substituting `path/to/code/` with the actual path leading to `GALA_main.py`:
   ```bash
   python ../path/to/code/GALA_main.py
   ```
   - The `..` implies that `GALA_main.py` is one directory level up from `MOF1`. Modify this according to your actual directory setup. You may call the absolute path of GALA_main.py as well.
4. Since the absolute path was left blank in `GALA.inp`, the script defaults to the current working directory for execution.
5. For multiple directories like `MOF1, MOF2, MOF3`, etc., utilize a `.sh` script to initiate separate jobs for each, harnessing the power of parallel processing.

### Important Tips
1. Always confirm your paths and directories are orderly to avoid errors and promote an efficient execution.
2. Monitor the output and logs to track the processing status and identify any potential issues requiring troubleshooting.
