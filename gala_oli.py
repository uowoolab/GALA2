#!/usr/bin/env python3

import os
import periodictable
from pymatgen.io.common import VolumetricData
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import iterate_structure


class GALA:
    def __init__(self, dir):
        """
        Initialize the GALA object by reading parameters from the 'GALA.inp' file.

        Parameters:
            dir: Directory where 'GALA.inp' is located
        """
        with open(f'{dir}/GALA.inp', 'r') as f:
            l = f.readlines()

        self.Method = l[27].upper().strip('\n')
        self.Directory = l[29].strip('\n')
        self.Temperature = float(l[33].strip('\n'))
        self.Cutoff_GCMC = float(l[35].strip('\n'))
        self.Delr = float(l[37].strip('\n'))
        self.Ewald = l[39].strip('\n')
        self.Grid_Factor = tuple(int(num)
                                 for num in l[41].strip('[]\n').replace(' ', ''))
        self.Grid_Spacing = float(l[43].strip('\n'))
        self.Sigma = float(l[47].strip('\n'))
        self.Radius = float(l[49].strip('\n'))
        self.Cutoff = float(l[51].strip('\n'))
        self.Write_Folded = l[53].strip('\n')
        self.Write_Smoothed = l[55].strip('\n')
        self.Selected_Guest = tuple(l[63].split())
        self.Selected_site = tuple([group.split(',') if ',' in group else [
            group] for group in l[65].split()])

    def check_guests_sites(self, fastmc):
        """
        Check if the guests and sites listed in GALA.inp are also present in the FastMC dictionary.

        Parameters:
            fastmc: FastMC object

        Returns:
            check_status: True if all guests and sites are found, False otherwise.
        """
        check_status = True
        missing_guests = [
            guest for guest in self.Selected_Guest if guest not in fastmc.guests]
        if missing_guests:
            print(
                f'Error: The following guests in the GALA dictionary are not found in the fastmc_dict: {missing_guests}')
            check_status = False

        fastmc_sites = [
            site for sublist in fastmc.guests_species for site in sublist]
        missing_sites = [
            site for site_group in self.Selected_site for site in site_group if site not in fastmc_sites]
        if missing_sites:
            print(
                f'Error: The following sites in the GALA dictionary are not found in the fastmc_dict: {missing_sites}')
            check_status = False

        return check_status


class FastMC:
    def __init__(self, fastmc_dir):
        """
        Initialize the FastMC object by reading data from the 'FIELD' file.

        Parameters:
            fastmc_dir: Directory where the 'FIELD' file is located
        """
        try:
            with open(os.path.join(fastmc_dir, "FIELD"), "r") as f:
                field_lines = f.readlines()
        except FileNotFoundError:
            print("Cannot find FIELD file...")

        self.name = field_lines[0].strip("\n")
        self.guests_name = []
        self.guests = []
        self.guests_species = []
        self.atomic_weight = []
        self.coordinates = []
        self.charge = []

        atom_count = 0
        read_atoms = False
        current_guest_atoms = []  # list to store current guest atoms
        current_guest_atom_type = []  # list to store current guest atoms type
        current_guest_atomic_weight = []  # list to store current guest atomic weights
        current_guest_coordinates = []  # list to store current guest coordinates
        current_guest_charge = []  # list to store current guest charge

        for line in field_lines:

            if "&guest" in line:
                if current_guest_atoms:
                    self.guests_species.append(current_guest_atoms)
                    self.guests.append(
                        ''.join(FastMC.get_molecule(current_guest_atom_type)))
                    self.atomic_weight.append(current_guest_atomic_weight)
                    self.coordinates.append(current_guest_coordinates)
                    self.charge.append(current_guest_charge)

                    current_guest_atoms = []
                    current_guest_atom_type = []
                    current_guest_atomic_weight = []
                    current_guest_coordinates = []
                    current_guest_charge = []

                guest_name = line.strip("&guest ").split(":")[0]
                self.guests_name.append(guest_name)
                read_atoms = True

            elif line.startswith('ATOMS') and read_atoms == True:
                atom_count = int(line.split()[1])

            elif atom_count > 0 and read_atoms == True:
                data = line.split()
                atom_label = data[0]
                atomic_weight = float(data[1])
                if atomic_weight != 0:
                    element = FastMC.get_periodic_element(atom_label)
                    if element:
                        current_guest_atom_type.append(element)
                    else:
                        pass

                x_coor = float(data[2])
                y_coor = float(data[3])
                z_coor = float(data[4])
                charge = float(data[5])

                current_guest_atoms.append(atom_label)
                current_guest_atomic_weight.append(atomic_weight)
                current_guest_coordinates.append((x_coor, y_coor, z_coor))
                current_guest_charge.append(charge)
                atom_count -= 1

            elif line.startswith('Framework'):
                read_atoms = False
                if current_guest_atoms:
                    self.guests_species.append(current_guest_atoms)
                    self.guests.append(
                        ''.join(FastMC.get_molecule(current_guest_atom_type)))
                    self.atomic_weight.append(current_guest_atomic_weight)
                    self.coordinates.append(current_guest_coordinates)
                    self.charge.append(current_guest_charge)

    @staticmethod
    def get_periodic_element(variable):
        """
        Parse a given variable and return the element symbol if found.

        Parameters:
            variable: Atom label from the 'FIELD' file

        Returns:
            symbol: Element symbol if found, None otherwise
        """
        element_symbols = [
            element.symbol for element in periodictable.elements]
        for symbol in element_symbols:
            if variable.startswith(symbol):
                return symbol
        return None

    @staticmethod
    def get_molecule(elements_list):
        """
        Count the number of each element in a list and return a list of molecules.

        Parameters:
            elements_list: List of elements (atom types)

        Returns:
            counted_molecule_list: List of molecules
        """
        counted_elements = []
        for element in elements_list:
            if element not in counted_elements:
                counted_elements.append(element)

        counted_molecule_list = []
        for element in counted_elements:
            count = elements_list.count(element)
            if count > 1:
                counted_molecule_list.append(f"{element}{count}")
            else:
                counted_molecule_list.append(element)

        return counted_molecule_list


class RASPA:
    """
        Initialize the RASPA object (Empty in your original code. You may add required functionalities).

    Parameters:
        raspa_dir: Directory where RASPA-related files are located
    """

    def __init__(self, raspa_dir):

        pass


class CubeFiles:
    """
    Initialize the CubeFiles object by reading data from .cube files.

    Parameters:
        dir: Directory where the .cube files are located
        guest: List of guest molecules
        site: List of guest sites
        fold: Grid factor for the GALA object
    """

    def __init__(self, dir, guest, site, fold):
        for guests, guest_sites in zip(guest, site):
            for sites in guest_sites:
                probability_file = f'{dir}/Prob_Guest[{guests}]_Site[{sites}]_folded.cube'
                probability_file_unfolded = f'{dir}/Prob_Guest[{guests}]_Site[{sites}].cube'
                if os.path.exists(probability_file):
                    self.cube = VolumetricData.from_cube(probability_file)
                    localdata = self.cube.data['total']
                    # Normalize the 3D array
                    localdata = localdata / np.sum(localdata)
                    self.datapoints = localdata
                elif not os.path.exists(probability_file) and os.path.exists(probability_file_unfolded):
                    self.cube = VolumetricData.from_cube(
                        probability_file_unfolded)
                    localdata = self.cube.data['total']
                    # Normalize the 3D array
                    localdata = localdata / np.sum(localdata)
                    localdata = localdata / float(fold[0]*fold[1]*fold[3])
                    self.datapoints = localdata
                else:
                    print(f'Error reading CUBE file: {probability_file}')

    def maxima(self, sigma, radius, cutoff):
        """
        Calculate the maxima of a 3D array (self.datapoints).

        Parameters:
            sigma: Value for the Gaussian filter
            radius: Radius for the maximum filter
            cutoff: Value to cutoff the maxima

        Returns:
            pruned_peaks: List of pruned peaks
        """
        original_data = self.datapoints
        temp_data = original_data
        normalising_sum = sum(temp_data)
        dimention = (np.array(self.cube.dim)).reshape(-1, 1)
        cell = self.cube.structure.lattice.matrix
        spacing = np.linalg.norm(cell[0][0] / dimention[0])
        cell_total = cell / dimention

        # compute the appropriate sigma and apply the gaussian filter
        sigma = (sigma/spacing)**0.5
        temp_data = gaussian_filter(temp_data, sigma, mode="wrap")

        # renormalise the data
        np.seterr(divide='ignore', invalid='ignore')
        temp_data *= normalising_sum/sum(temp_data)

        # generate the binary structure for the neighborhood
        neighborhood = generate_binary_structure(np.ndim(temp_data), 2)

        # calculate the footprint and iterate the structure
        footprint = int(round(radius / spacing, 0))
        neighborhood = iterate_structure(neighborhood, footprint)

        # apply the local maximum filter
        local_max = maximum_filter(
            temp_data, footprint=neighborhood, mode='wrap') == temp_data

        # subtract the background from the mask
        background = (temp_data == 0)
        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)
        detected_peaks = local_max ^ eroded_background

        peaks = np.where(detected_peaks)
        cartesian_peaks = []
        for point in zip(peaks[0], peaks[1], peaks[2]):
            if np.all(temp_data[point] > 0.0):
                cartesian_peaks.append(
                    (np.dot(point, cell_total).tolist(), temp_data[point]))

        # prune the peaks below the cutoff
        pruned_peaks = []
        previous_value = 0.0
        maximum_value = max([peak[1] for peak in cartesian_peaks])
        for point in sorted(cartesian_peaks, key=lambda k: -k[1]):
            if point[1] > cutoff*maximum_value:
                previous_value = point[1]
                pruned_peaks.append(point)
            else:
                break

        return pruned_peaks


if __name__ == "__main__":
    """
    Main execution of the script. 
    Creates instances of GALA and FastMC or RASPA based on the method used. 
    Then it loops through all the guests and sites to compute the maxima of a 3D array for each combination.
    Finally, it prints the combined output.
    """
    GALA_MAIN = '/Users/oliviermarchand/Desktop/Python_Codes/Masters/GALA_v2/Jake/GALA-main'

    gala = GALA(GALA_MAIN)

    if gala.Method == 'FASTMC':
        fastmc = FastMC(gala.Directory)
    elif gala.Method == 'RASPA':
        raspa = RASPA(gala.Directory)
    else:
        print('Unsupported Method: Error, line 2 GALA.inp')

    # Create a list to store processed combinations
    processed_combinations = []
    # At the start of your script, before the loop
    combined_output = []

    for guest_index, guest in enumerate(fastmc.guests):
        if guest in gala.Selected_Guest:
            sites = fastmc.guests_species[guest_index]
            for site in sites:
                combination = (guest, site)

                # Check if this combination has already been processed
                if combination not in processed_combinations:
                    cube_files = CubeFiles(gala.Directory, [guest], [
                                           [site]], gala.Grid_Factor)
                    peaks = cube_files.maxima(
                        gala.Sigma, gala.Radius, gala.Cutoff)

                    # Append the combination and peaks to your output
                    for peak in peaks:
                        combined_output.append((guest, site, peak))

                    # After processing the combination, add it to the list
                    processed_combinations.append(combination)
                else:
                    pass
                    # print(f"Combination {combination} already processed.")

    print(np.matrix(combined_output))
