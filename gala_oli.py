#!/usr/bin/env python3

import os
import numpy as np
from pymatgen.core import Element, IMolecule, Molecule
from pymatgen.io.common import VolumetricData
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion, iterate_structure


class GalaInput:
    def __init__(self, dir):
        """
        Initialize the GalaInput class with the provided directory.

        Args:
            dir (str): Directory path containing GALA input files.
        """
        with open(f'{dir}/GALA.inp', 'r') as f:
            lines = f.readlines()

        self.Method = lines[27].upper().strip('\n')
        self.Directory = lines[29].strip('\n')
        self.Temperature = float(lines[33].strip('\n'))
        self.Cutoff_GCMC = float(lines[35].strip('\n'))
        self.Delr = float(lines[37].strip('\n'))
        self.Ewald = lines[39].strip('\n')
        self.Grid_Factor = tuple(int(num)
                                 for num in lines[41].strip('[]\n').replace(' ', ''))
        self.Grid_Spacing = float(lines[43].strip('\n'))
        self.Sigma = float(lines[47].strip('\n'))
        self.Radius = float(lines[49].strip('\n'))
        self.Cutoff = float(lines[51].strip('\n'))
        self.Write_Folded = lines[53].strip('\n')
        self.Write_Smoothed = lines[55].strip('\n')
        self.Selected_Guest = tuple(lines[63].split())
        self.Selected_Site = tuple([group.split(',') if ',' in group else [
            group] for group in lines[65].split()])


class GuestStructure:
    def __init__(self, gala):
        """
        Initialize the GuestStructure class with the specified method, directory, and GALA instance.

        Args:
            gala (GalaInput): Instance of the GalaInput class.
        """
        self.gala = gala
        self.method = self.gala.Method
        self.directory = self.gala.Directory

        if self.method == 'FASTMC':
            guests_chemical_structure, guests_atoms_coordinates, guest_sites_labels, site_data = self.parse_fastmc()
            self.guest_molecules = []

            for element, coordinate, molecule_site_data in zip(guests_chemical_structure, guests_atoms_coordinates, site_data):
                guest_molecule = GuestMolecule(
                    species=element, coords=coordinate, sites_data=molecule_site_data, gala_instance=self.gala)
                self.guest_molecules.append(guest_molecule)

    def __str__(self):
        """
        Return a string representation of the GuestStructure object.

        Returns:
            str: String representation of the GuestStructure object.
        """
        molecule_strings = [str(molecule) for molecule in self.guest_molecules]
        return "\n".join(molecule_strings)

    def parse_fastmc(self):
        """
        Parse the FASTMC guest file in the specified directory.

        Returns:
            tuple: Tuple containing parsed guest information (elements, coordinates, site_labels, site_data).
        """

        if len(self.gala.Selected_Guest) != len(self.gala.Selected_Site):
            raise ValueError(
                "Error Code: GUEST_SITE_MISMATCH\nError Description: The selected guest and site information is inconsistent.")

        try:
            with open(os.path.join(self.gala.Directory, "FIELD"), "r") as f:
                field_lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Cannot find FIELD file in directory:", self.gala.Directory)

        guests = []

        atom_count = 0
        read_atoms = False
        def reset_current_guest(): return {'valid_labels': [], 'coordinates': [
        ], 'species': [], 'site_data': []}
        current_guest = reset_current_guest()

        for line in field_lines:
            if "&guest" in line:
                read_atoms = True
                if current_guest['valid_labels']:
                    guests.append(current_guest)
                    current_guest = reset_current_guest()

            elif line.startswith('ATOMS') and read_atoms:
                atom_count = int(line.split()[1])

            elif atom_count > 0 and read_atoms:
                atom_label, atom, coordinate, charge = self.process_atom(line)
                if atom != 'D':
                    current_guest['valid_labels'].append(atom)
                    current_guest['coordinates'].append(coordinate)
                else:
                    pass
                current_guest['species'].append(atom_label)
                current_guest['site_data'].append(
                    [atom_label, atom, coordinate, charge])
                atom_count -= 1

            elif line.startswith('Framework'):
                read_atoms = False
                if current_guest['valid_labels']:
                    guests.append(current_guest)
                    current_guest = reset_current_guest()

        guests_reduced, elements, dummy_atoms = self.post_process_guests(
            guests)

        
        selected_guests = self.select_guests(
            guests, guests_reduced, elements, dummy_atoms)

        return selected_guests

    def process_atom(self, line):
        """
        Process an atom line from the FIELD file and extract relevant information.

        Args:
            line (str): Atom line from the FIELD file.

        Returns:
            tuple: Tuple containing the atom label, element, coordinates, and charge.
        """
        data = line.split()
        atom_label = data[0]
        atomic_weight = float(data[1])
        x_coor = float(data[2])
        y_coor = float(data[3])
        z_coor = float(data[4])
        charge = float(data[5])
        coordinate = (x_coor, y_coor, z_coor)

        if atomic_weight == 0.0:
            atom = 'D'
        else:
            atom = self.find_closest_element(atomic_weight)

        return atom_label, atom, coordinate, charge

    def find_closest_element(self, atomic_weight):
        """
        Find the closest chemical element to the given atomic weight.

        Args:
            atomic_weight (float): Atomic weight of the element.

        Returns:
            str: Symbol of the closest chemical element.
        """
        min_diff = float("inf")
        closest_element = None
        for element in Element:
            diff = abs(element.atomic_mass - atomic_weight)
            if diff < min_diff:
                min_diff = diff
                closest_element = element.symbol
        return closest_element

    def post_process_guests(self, guests):
        """
        Perform post-processing on the parsed guests.

        Args:
            guests (list): List of parsed guest dictionaries.

        Returns:
            tuple: Tuple containing the reduced formulas of the guests, the list of chemical elements, and the set of dummy atoms.
        """
        guests_reduced = []
        for guest in guests:
            guests_reduced.append(Molecule(
                guest['valid_labels'], guest['coordinates']).composition.reduced_formula)

        elements = [str(e) for e in Element]
        dummy_atoms = set(
            species for guest in guests for species in guest['species'] if species not in elements)
        return guests_reduced, elements, dummy_atoms

    def select_guests(self, guests, guests_reduced, elements, dummy_atoms):
        """
        Select the guests based on the selected guest and site information.

        Args:
            guests (list): List of parsed guest dictionaries.
            guests_reduced (list): List of reduced formulas of the guests.
            elements (list): List of chemical elements.
            dummy_atoms (set): Set of dummy atoms.

        Returns:
            tuple: Tuple containing the valid labels, coordinates, species, and site data of the selected guests.
        """
        selected_guest_indices = [guests_reduced.index(
            guest) for guest in self.gala.Selected_Guest]
        valid_labels = []
        coordinates = []
        species = []
        site_data = []

        for i in selected_guest_indices:
            selected_guest = {k: guests[i][k] for k in (
                'valid_labels', 'coordinates', 'species', 'site_data')}
            selected_guest['site_data'] = self.filter_site_data(
                selected_guest['site_data'], i, elements, dummy_atoms)
            valid_labels.append(selected_guest['valid_labels'])
            coordinates.append(selected_guest['coordinates'])
            species.append(selected_guest['species'])
            site_data.append(selected_guest['site_data'])

        return valid_labels, coordinates, species, site_data

    def filter_site_data(self, site_data, i, elements, dummy_atoms):
        """
        Filter the site data based on the selected sites.

        Args:
            site_data (list): List of site data.
            i (int): Index of the selected guest.
            elements (list): List of chemical elements.
            dummy_atoms (set): Set of dummy atoms.

        Returns:
            list: Filtered site data.
        """
        filtered_site_data = [data for data in site_data if data[0] in self.gala.Selected_Site[i]
                              or data[1] not in elements and any(dummy in self.gala.Selected_Site[i] for dummy in dummy_atoms)]
        return self.group_site_data(filtered_site_data)

    def group_site_data(self, filtered_site_data):
        """
        Group the site data based on the labels.

        Args:
            filtered_site_data (list): List of filtered site data.

        Returns:
            list: Grouped site data.
        """
        grouped_site_data = {}
        for data in filtered_site_data:
            label = data[0]
            if label not in grouped_site_data:
                grouped_site_data[label] = {
                    'element': data[1], 'coordinates': [], 'charges': []}
            grouped_site_data[label]['coordinates'].append(data[2])
            grouped_site_data[label]['charges'].append(data[3])
        return [[label, data['element'], data['coordinates'], data['charges']] for label, data in grouped_site_data.items()]


class GuestMolecule(IMolecule):
    def __init__(self, species, coords, sites_data, gala_instance):
        """
        Initialize the GuestMolecule class with the specified species, coordinates, sites_data, and GALA instance.

        Args:
            species (str): Chemical species of the molecule.
            coords (list): Coordinates of the molecule.
            sites_data (list): Site data of the molecule.
            gala_instance (GalaInput): Instance of the GalaInput class.
        """
        super().__init__(species, coords)
        self.gala = gala_instance
        self.guest_sites = [Site(*site_data, parent_molecule=self)
                            for site_data in sites_data]

    def __str__(self):
        """
        Return a string representation of the GuestMolecule object.

        Returns:
            str: String representation of the GuestMolecule object.
        """
        return "Molecule: {}\nMolecule Coordinates:\n{}\nSites:\n{}".format(
            self.composition.reduced_formula,
            np.array(self.cart_coords),
            '\n'.join(f"Site {i+1}: {sites}" for i, sites in enumerate(map(str, self.guest_sites))))


class Site:
    def __init__(self, label, element, coords, charge, parent_molecule):
        """
        Initialize the Site class with the specified label, element, coordinates, charge, and parent molecule.

        Args:
            label (str): Label of the site.
            element (str): Chemical element of the site.
            coords (tuple): Coordinates of the site.
            charge (float): Charge of the site.
            parent_molecule (GuestMolecule): Parent molecule of the site.
        """
        self.label = label
        self.element = element
        self.coords = coords
        self.charge = charge
        self.parent_molecule = parent_molecule
        self.gala = parent_molecule.gala
        self.cube = None
        self._datapoints = None
        self._maxima = None
        self._maxima_coordinates = None
        self._maxima_values = None

    def __str__(self):
        """
        Return a string representation of the Site object.

        Returns:
            str: String representation of the Site object.
        """
        maxima_info = self.maxima
        return f"Site Label: {self.label}, Element: {self.element}, Coords: {self.coords}, Charge: {self.charge}\n{maxima_info}"

    @property
    def maxima(self):
        """
        Calculate and return the maxima information of the site.

        Returns:
            tuple: Tuple containing the fractional coordinates of the maxima and the maxima values.
        """
        if self.cube is None or self._datapoints is None:
            self.load_cube_data()

        if self._maxima_coordinates is None or self._maxima_values is None:
            self.calculate_maxima()

        maxima_coords, maxima_values = self._maxima_coordinates, self._maxima_values

        if maxima_coords is not None:
            fractional_coords = self.maxima_fractional_coordinates
        else:
            raise Exception(
                'Maxima coordinates not calculated. Run calculate_maxima first.')

        maxima_info = "Maxima:\n"
        for i, (coords, value) in enumerate(zip(maxima_coords, maxima_values), start=1):
            maxima_info += f"\tMaxima {i}: Value: {value}\n\t\t  Cartesian Coordinates: {list(coords)}\n\t\t  Fractional Coordinates: {list(fractional_coords[i-1])}\n"
        
        return maxima_info

        # return maxima_values

    @property
    def maxima_fractional_coordinates(self):
        """
        Calculate and return the fractional coordinates of the maxima.

        Returns:
            list: Fractional coordinates of the maxima.
        """
        if self._maxima_coordinates is None:
            raise Exception(
                'Maxima coordinates not calculated. Run calculate_maxima first.')

        if self.cube is None:
            raise Exception('Cube data not loaded. Run load_cube_data first.')

        lattice = self.cube.structure.lattice

        fractional_coordinates = [lattice.get_fractional_coords(
            cart_coords) for cart_coords in self._maxima_coordinates]

        return fractional_coordinates

    def load_cube_data(self):
        """
        Load the cube data for the site.
        """
        dir = self.gala.Directory
        guests = self.parent_molecule.composition.reduced_formula
        sites = self.label
        fold = self.gala.Grid_Factor

        probability_file = f'{dir}/Prob_Guest[{guests}]_Site[{sites}]_folded.cube'
        probability_file_unfolded = f'{dir}/Prob_Guest[{guests}]_Site[{sites}].cube'

        if os.path.exists(probability_file):
            try:
                self.cube = VolumetricData.from_cube(probability_file)
                localdata = self.cube.data['total']
                localdata = localdata / np.sum(localdata)
                self._datapoints = localdata
            except Exception as e:
                raise Exception(f'Error loading cube file: {probability_file}')

        elif not os.path.exists(probability_file) and os.path.exists(probability_file_unfolded):
            try:
                self.cube = VolumetricData.from_cube(probability_file_unfolded)
                localdata = self.cube.data['total']
                localdata = localdata / np.sum(localdata)
                localdata = localdata / float(fold[0]*fold[1]*fold[3])
                self._datapoints = localdata
            except Exception as e:
                raise Exception(
                    f'Error loading cube file: {probability_file_unfolded}')

        else:
            raise FileNotFoundError('''Error Code: UNAVAILABLE_CUBE_FILE
Error Description: Unavailable Probability File
Error Message: We apologize, but we couldn't locate any available unfolded or folded probability file required for this operation.\n''')

    def calculate_maxima(self):
        """
        Calculate the maxima for the site.
        """
        if self._datapoints is None or self.cube is None:
            raise Exception('Data not loaded. Run load_cube_data first.')

        original_data = self._datapoints
        temp_data = original_data
        normalising_sum = sum(temp_data)
        dimension = (np.array(self.cube.dim)).reshape(-1, 1)
        cell = self.cube.structure.lattice.matrix
        spacing = np.linalg.norm(cell[0][0] / dimension[0])
        cell_total = cell / dimension

        sigma = (self.gala.Sigma / spacing) ** 0.5
        temp_data = gaussian_filter(temp_data, sigma, mode="wrap")

        temp_data *= normalising_sum / sum(temp_data)

        neighborhood = generate_binary_structure(np.ndim(temp_data), 2)

        footprint = int(round(self.gala.Radius / spacing, 0))
        neighborhood = iterate_structure(neighborhood, footprint)

        local_max = maximum_filter(
            temp_data, footprint=neighborhood, mode='wrap') == temp_data

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

        pruned_peaks = []
        maximum_value = max([peak[1] for peak in cartesian_peaks])
        for point in sorted(cartesian_peaks, key=lambda k: -k[1], reverse=True):
            if point[1] > self.gala.Cutoff * maximum_value:
                pruned_peaks.append(point)

        self._maxima_coordinates = [point[0] for point in pruned_peaks]
        self._maxima_values = [point[1] for point in pruned_peaks]


if __name__ == "__main__":
    GALA_MAIN = os.getcwd()
    gala_input = GalaInput(GALA_MAIN)

    # Print all output for specifed Guests and Sites
    structure = GuestStructure(gala_input)
    print(structure)

    # Prints only maxima for guest N2 of Nx site
    print(structure.guest_molecules[1].guest_sites[0].maxima)

    # Print IMolecule center of mass of guest 0 (CO2)
    print(structure.guest_molecules[0].center_of_mass)

    # Print IMolecule centered molecule of guest 0 (N2)
    print(structure.guest_molecules[1].get_centered_molecule)

    # Print Specific Site Data, for example guest 0, site 1 (CO2, Cx)
    print(structure.guest_molecules[0].guest_sites[0])

    # Print Specific Site Data, for example guest 0, site 1 (N2, COM (D))
    print(structure.guest_molecules[1].guest_sites[1])

    # You can also remove Guest or Sites in the input file and the print(structure) will only print the guest and sites you specified
