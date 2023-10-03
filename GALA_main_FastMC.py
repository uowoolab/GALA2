#!/usr/bin/env python3

import os
import numpy as np
from pymatgen.core import Element, IMolecule, Molecule, Lattice, Structure
from pymatgen.io.common import VolumetricData
from pymatgen.util.coord import all_distances
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion, iterate_structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from itertools import combinations
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.io.cif import CifWriter
import shutil
import subprocess

import time

import multiprocessing

# Comments Oli
# I removed the RASPA functions from this version to limit the length of the code.
# This code will take the FASTmc field file, and the probability plots for each sites in the guest.
# I modified the naming scheme for the probability plots to that it works on wookie:
#   >> The naming scheme for the output file is as followed:
# > Prob_Guest_X_Site_Y_folded.cube
# > X == Guest, Y == Site
# To run on wookie I use the 'newgala' env I compiled which is found on my scratch
# I also module load dl_poly (default dl_poly 1.10)

# TODO Add a log file with key checkpoints?
# TODO Add code to fold and change probability file names
# TODO Prunning based off the dl poly data
# TODO Prunning based off symmetry (before dl poly calculations would be better to shorten the length of the code)
# TODO Review the standard dl poly folders might only need to run a singular calculation
# Try TODO implementation of a molecule property for pymatgen (instead of only site_properties)
# TODO Update pymatgen's supercell structure
# TODO Optimization (Not sure yet how, but further testing will narrow to sections)
# TODO Modularization, will have to wait until code is done, too many big changes
# TODO Return an output file contaning Structure information, guest information, binding sites, binding site energies, maxima peaks/coords.
# TODO Implement more user options, such as allowing them to name their own folders, wether they want only one output file or all the "temp" files. More will come to mind later on.
# TODO Make sure we save output files in wanted directory
# TODO Make sure class, methods and variable names are adequate and consistant. 
# For example, cell is it cell vectors (3x3 matrix), cell scalars/size (4,3,3), unit cell or supercell vectors, etc (was quite confusing when trying to read through faps and old gala)
# TODO Expend on guest compatibility
# TODO TEST with different input files, guests, etc

# TODO No-folded FIELD convert to folded FIELD file, if they want to fold they will require to specify the grid factor

# TODO Better docstring, 
# TODO Set up how to use documentation
# ---
# TODO for Jake

# ---

# Done list
# ///////////////////////////////////////////////////////////////////////////
# TODO Add fix from other code for when dummy atoms have no probability plots
# TODO Rework the Maxima, if not plot given for dummy atom no problem, nut must need plot for atomic sites
# TODO Add executable absolute path to DL POLY in .inp file 
# TODO Use cutoff to find minimal grid factor and use that to make supercell for dl poly
# TODO Grid factor should be computed for the cutoff they want
# TODO Reorder binding sites from highest occupencies to lowest in xyz
# TODO normalize each occupencies to 1 so that it is a % of the total occupencies
# TODO Check to add dl poly output to see if something went wrong
# TODO Write error instead of print, check if first dl poly ran, (STASIS) if not kill job
# TODO Max Number of binding site
# TODO Default cwd directry
# TODO Default fold factor (1,1,1)
# TODO clean up github for public and group members
# ///////////////////////////////////////////////////////////////////////////


class GalaInput:
    def __init__(self, dir):
        """
        Initialize the GalaInput class with the provided directory.

        Args:
            dir (str): Directory path containing GALA input files.
        """
        with open(f'{dir}/GALA.inp', 'r') as f:
            lines = f.readlines()

        self.directory = dir

        # General Input
        self.method = self._get_method(lines[27])
        self.directory = self._get_directory(lines[29])
        self.max_number_bs = self._get_max_number_bs(lines[31])
        self.output_directory = self._get_output_directory()

        # MD Parameters
        self.md_exe = lines[35].strip('\n')
        self.md_cutoff = float(lines[37].strip('\n'))
        self.md_cpu = self._get_md_cpu(lines[39])

        # GCMC Analysis Parameters
        self._set_gcmc_parameters(lines)

        # Probability Plot Guest and Site Selection
        self.selected_guest = tuple(lines[62].split())
        self.selected_site = self._get_selected_site(lines[64])

        # Post Process Cleaning Section
        self.cleaning = lines[68].strip('\n').upper() == "T"

    def _get_method(self, line):
        return line.upper().strip('\n')

    def _get_directory(self, line):
        directory = line.strip('\n')
        return directory if directory else self.directory

    def _get_max_number_bs(self, line):
        try:
            max_number_bs = int(line.strip('\n'))
            if max_number_bs > 0:
                return max_number_bs
            raise ValueError
        except ValueError:
            print('Warning: Invalid input. All binding sites will be printed.')
            return float('inf')

    def _get_output_directory(self):
        output_directory = os.path.join(self.directory, 'GALA_Output')
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        return output_directory

    def _get_md_cpu(self, line):
        try:
            md_cpu = int(line.strip('\n'))
            if md_cpu >= 1:
                return md_cpu
            raise ValueError
        except ValueError:
            print('Using default CPU allocation for DL Poly calculations')
            return 1

    def _set_gcmc_parameters(self, lines):
        self.gcmc_sigma = float(lines[43].strip('\n'))
        self.gcmc_radius = float(lines[45].strip('\n'))
        self.gcmc_cutoff = float(lines[47].strip('\n'))
        self.gcmc_write_folded = lines[49].strip('\n').upper() == "T"
        self.gcmc_grid_factor = self._get_gcmc_grid_factor(lines[51])
        print(self.gcmc_grid_factor)
        self.gcmc_write_smoothed = lines[53].strip('\n').upper() == "T"

    def _get_gcmc_grid_factor(self, line):
        try:
            grid_factor = tuple(int(num) for num in line.strip('[]\n').split())
            if grid_factor:
                return grid_factor
            raise ValueError
        except ValueError:
            print('Using default grid factor (1,1,1)')
            return 1, 1, 1

    def _get_selected_site(self, line):
        return tuple([group.split(',') if ',' in group else [group] for group in line.split()])


class GuestStructure:
    def __init__(self, gala):
        """
        Initialize the GuestStructure class with the specified method, directory, and GALA instance.

        Args:
            gala (GalaInput): Instance of the GalaInput class.
        """
        self.gala = gala
        self.method = self.gala.method
        self.directory = self.gala.directory
        self._structure_name = None

        if self.method == 'FASTMC':
            guests_chemical_structure, guests_atoms_coordinates, guest_sites_labels, site_data_unfilted, site_data = self.parse_fastmc()

            self.guest_molecules = []

            for element, sites_labels, coordinate, molecule_site_data_unfilted, molecule_site_data in zip(guests_chemical_structure, guest_sites_labels, guests_atoms_coordinates, site_data_unfilted, site_data):
                guest_molecule = GuestMolecule(
                    species=element, sites_labels_save=sites_labels, coords=coordinate, sites_data_unfilted=molecule_site_data_unfilted, sites_data=molecule_site_data, structure_name=self._structure_name, gala_instance=self.gala)
                self.guest_molecules.append(guest_molecule)

        else:
            raise ValueError(f"Unknown method: {self.method}")

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

        if len(self.gala.selected_guest) != len(self.gala.selected_site):
            raise ValueError(
                "Error Code: GUEST_SITE_MISMATCH\nError Description: The selected guest and site information is inconsistent.")

        try:
            with open(os.path.join(self.gala.directory, "FIELD"), "r") as f:
                field_lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Cannot find FIELD file in directory:", self.gala.directory)

        self._structure_name = field_lines[0].replace('\n', '')

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
        charge = float(data[2])
        x_coor = float(data[3])
        y_coor = float(data[4])
        z_coor = float(data[5])
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
            guest) for guest in self.gala.selected_guest]
        valid_labels = []
        coordinates = []
        species = []
        site_data = []
        site_data_unfilted = []

        for i in selected_guest_indices:
            selected_guest = {k: guests[i][k] for k in (
                'valid_labels', 'coordinates', 'species', 'site_data')}
            selected_guest['site_data_unfilted'], selected_guest['site_data'] = self.filter_site_data(
                selected_guest['site_data'], i, elements, dummy_atoms)
            valid_labels.append(selected_guest['valid_labels'])
            coordinates.append(selected_guest['coordinates'])
            species.append(selected_guest['species'])
            site_data_unfilted.append(selected_guest['site_data_unfilted'])
            site_data.append(selected_guest['site_data'])

        return valid_labels, coordinates, species, site_data_unfilted, site_data

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

        filtered_site_data = [data for data in site_data if data[0] in self.gala.selected_site[i]
                              or data[1] not in elements and any(dummy in self.gala.selected_site[i] for dummy in dummy_atoms)]

        return filtered_site_data, self.group_site_data(filtered_site_data)

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


class GuestMolecule(Molecule):
    def __init__(self, species, sites_labels_save, coords, sites_data_unfilted=None, sites_data=None, structure_name=None, gala_instance=None, charge=None, spin_multiplicity=None, site_properties=None, charge_spin_check=None):
        """
        Initialize the GuestMolecule class with the specified species, coordinates, sites_data, and GALA instance.

        Args:
            species (str): Chemical species of the molecule.
            coords (list): Coordinates of the molecule.
            sites_data (list): Guest_Sites data of the molecule.
            gala_instance (GalaInput): Instance of the GalaInput class.
        """
        super().__init__(species, coords)
        self.guest_molecule_data = Molecule(
            species=[i[1] for i in sites_data_unfilted],
            coords=[i[2] for i in sites_data_unfilted],
            labels=[i[0] for i in sites_data_unfilted],
            site_properties={'elements': [i[1] for i in sites_data_unfilted], 'charges': [
                i[3] for i in sites_data_unfilted]}
        )
        self.structure_name = structure_name
        self.gala = gala_instance
        self.guest_sites = None
        self.sites_labels_save = sites_labels_save
        if sites_data is not None:
            self.guest_sites = [GuestSites(*site_data, parent_molecule=self)
                                for site_data in sites_data]

        self._binding_sites = None
        self._binding_sites_cart = None
        self._binding_site_maxima = None
        self.structure_with_sites = None
        self._structure_match = None

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

    @property
    def binding_sites(self):
        """
        Property method to access the calculated binding sites. If the binding sites have not been calculated yet, it calls
        the calculate_binding_sites() method to calculate them first. It returns the coordinates of the binding sites.

        Returns:
            list: A list of binding site coordinates.
        """
        if self._binding_sites == None:
            self.calculate_binding_sites()

        binding_site_coords = self._binding_sites
        return binding_site_coords

    @property
    def get_cube_structure(self):
        """
        Retrieve the structure of the first guest site from the list of guest sites. It ensures that all structures match.

        Returns:
            Structure: The structure object.
        """
        if not self.guest_sites:
            raise Exception("No guest sites found")

        if self._structure_match is None:
            self.check_structure_match()

        return self.guest_sites[0].structure

    def check_structure_match(self):
        """
        Compare the structures of all guest sites to ensure that they match, and raises an exception if they don't.

        Raises:
            Exception: If structures in the cube files do not match.
        """
        first_structure = self.guest_sites[0].structure

        matcher = StructureMatcher()

        for site in self.guest_sites[1:]:
            if not matcher.fit(first_structure, site.structure):
                self._structure_match = False
                raise Exception("Structures in the cube files do not match!")

        self._structure_match = True

    def calculate_binding_sites(self):
        """
        Calculate the binding sites for the guest molecules around the central molecule in the structure. It first calculates
        the distance of each atom from the center of mass. Then, it sorts the atoms based on their distances from shortest to
        longest. Next, it calculates the distance between each pair of atoms and stores the distances in a dictionary.
        Finally, it determines the binding sites by considering the distances and overlaps between atoms and stores the result
        in the _binding_sites attribute.
        """

        get_struct = self.get_cube_structure

        guest_atom_distances = []

        # Calculate the distance of each atom from the center of mass
        for idx, atom in enumerate(self.guest_molecule_data.site_properties['elements']):
            dist = np.linalg.norm(self.center_of_mass -
                                  self.guest_molecule_data.cart_coords[idx])
            for site in self.guest_sites:
                if atom != 'D':
                    if str(atom) == str(site.element):
                        guest_atom_distances.append((dist, idx, site.label))
                else:
                    pass

        # Sort the atoms based on the distance from shortest to longest
        guest_atom_distances.sort()

        # Replace old index with new one based on sorted order
        guest_atom_distances = [(dist, idx_new, atom) for idx_new,
                                (dist, idx_old, atom) in enumerate(guest_atom_distances)]

        # Calculate distance between each pair of atoms
        distances = {}
        for (dist1, idx1, atom1), (dist2, idx2, atom2) in combinations(guest_atom_distances, 2):
            pair_dist = np.linalg.norm(
                abs(guest_atom_distances[idx1][0]) + abs(guest_atom_distances[idx2][0]))
            distances[f'{idx1}_{idx2}'] = pair_dist

        overlap_tol = 0.3
        binding_sites = []
        occupancies = []
        cleaned_binding_site = []

        # Atom closest to the COM (e.g., Cx or Nx)
        origin_key = guest_atom_distances[0][2]

        # Working with a dict of {label: site}
        sites = {x.label: x for x in self.guest_sites}

        for central_max, central_max_value in zip(sites[origin_key].maxima_cartesian_coordinates,
                                                  sites[origin_key]._maxima_values):

            # Single site guest, just place it at the maximum
            if '0_1' not in distances:
                binding_sites.append(
                    [(guest_atom_distances[0][1], central_max, central_max_value)])
                continue

            # If we have more than one atom to align
            align_key = guest_atom_distances[1][2]
            # Why are we choosing 999.9 as the comparison point lol?
            align_closest = (999.9, None)
            align_found = False
            for align_atom_max, align_atom_max_value in zip(sites[align_key].maxima_cartesian_coordinates,
                                                            sites[align_key]._maxima_values):
                if np.all(align_atom_max == central_max):
                    continue
                # Distance vector and distance between central site and alignment site
                vector_0_1 = pbc_shortest_vectors(get_struct.lattice,
                                                  get_struct.lattice.get_fractional_coords(
                                                      central_max),
                                                  get_struct.lattice.get_fractional_coords(align_atom_max))[0][0]

                separation_0_1 = np.linalg.norm(vector_0_1)

                # How much overlap is there between the sites according to plot vs expected
                align_overlap = abs(separation_0_1 - distances['0_1'])

                if align_overlap < align_closest[0]:

                    align_closest = (align_overlap, align_atom_max)
                if align_overlap < overlap_tol:
                    align_found = True

                    # If we only have a two atom guest
                    if '0_2' not in distances:

                        binding_sites.append([
                            (guest_atom_distances[0][1],
                             central_max, central_max_value),
                            (guest_atom_distances[1][1], vector_0_1)
                        ])

                        continue

                    # If we have three sites
                    # IMPORTANT NOTE (Jake): I'm changing the first index to 2 instead of 1... otherwise it uses the same site
                    # as the orient_key. This doesn't matter for things like CO2 or H2O, but very important for something like HCN (three unique atoms)
                    # In the old code, it was [1][2] instead of [2][2], but I think this is a bug
                    orient_key = guest_atom_distances[2][2]
                    orient_closest = (999.9, None)
                    found_site = False
                    for orient_atom_max, orient_atom_max_value in zip(sites[orient_key].maxima_cartesian_coordinates,
                                                                      sites[orient_key]._maxima_values):

                        vector_0_2 = pbc_shortest_vectors(get_struct.lattice,
                                                          get_struct.lattice.get_fractional_coords(
                                                              central_max),
                                                          get_struct.lattice.get_fractional_coords(orient_atom_max))[0][0]
                        separation_0_2 = np.linalg.norm(vector_0_2)
                        vector_1_2 = pbc_shortest_vectors(get_struct.lattice,
                                                          get_struct.lattice.get_fractional_coords(
                                                              align_atom_max),
                                                          get_struct.lattice.get_fractional_coords(orient_atom_max))[0][0]

                        separation_1_2 = np.linalg.norm(vector_1_2)

                        overlap_0_2 = abs(separation_0_2 - distances['0_2'])
                        overlap_1_2 = abs(separation_1_2 - distances['1_2'])

                        # This is the new closest orientation
                        if overlap_0_2 + 0.5*overlap_1_2 < orient_closest[0]:
                            orient_closest = (
                                overlap_0_2 + 0.5*overlap_1_2, orient_atom_max)

                        # If we find two aligning sites within tolerance (multiple of 2 since we are fitting two sites)
                        if overlap_0_2 < overlap_tol and overlap_1_2 < 2*overlap_tol:
                            found_site = True
                            # Add all three sites
                            binding_sites.append([
                                (guest_atom_distances[0][1],
                                    central_max,
                                    central_max_value),
                                (guest_atom_distances[1][1], vector_0_1),
                                (guest_atom_distances[2][1], vector_0_2)])

                    if not found_site:
                        binding_sites.append([
                            (guest_atom_distances[0][1],
                                central_max,
                                central_max_value),
                            (guest_atom_distances[1][1], vector_0_1),
                            (guest_atom_distances[2][1], vector_0_2)])

                else:
                    if '0_2' not in distances and align_closest[0] > distances['0_1']:
                        # Very isolated atom, not within 2 distances of any others
                        # treat as isolated point atom and still make a guest
                        binding_sites.append(
                            [(guest_atom_distances[0][1], central_max, central_max_value)])
       
        binding_sites = self.remove_duplicates_and_inverses(binding_sites)
        binding_sites = sorted(binding_sites, key=lambda x: x[0][2], reverse=True)

        if self.gala.max_number_bs != float('inf'):
            if len(binding_sites) > self.gala.max_number_bs:
                binding_sites = binding_sites[:self.gala.max_number_bs]
            elif len(binding_sites) < self.gala.max_number_bs:
                print("Less binding sites found than requested, proceeding with all binding sites")

        for site in binding_sites:
            occupancy = site[0][2]
            occupancies.append(occupancy)

            include_guests = [
                [str(element.symbol) for element in self.guest_molecule_data.species], [str(element) for element in self.guest_molecule_data.labels], self.aligned_to(*site)]
            cleaned_binding_site.append(include_guests)
   
        self.guest_molecule_data.add_site_property('elemental_binding_site', [list(
            row) for row in zip(*[i[2] for i in cleaned_binding_site])])

        self._binding_site_maxima = binding_sites
        self._binding_sites = self.remove_duplicates(cleaned_binding_site)
        self._binding_sites_cart = self.convert_to_cartesian(
            cleaned_binding_site)

    def convert_to_cartesian(self, input_data):
        """
        Converts fractional coordinates to Cartesian coordinates for a list of molecules.

        Args:
            self (object): The object calling this method.
            input_data (list): A list of molecules, where each molecule is represented as a list
                containing three elements: atom types, atom labels, and fractional positions.

        Returns:
            list: A list of molecules, where each molecule is represented as a list containing
                three elements: atom types, atom labels, and Cartesian positions.
        """
        get_struct = self.get_cube_structure

        cartesian_data = []

        for molecule in input_data:
            atom_types = molecule[0]
            atom_label = molecule[1]
            fractional_positions = molecule[2]

            cartesian_positions = [get_struct.lattice.get_cartesian_coords(
                pos) for pos in fractional_positions]

            cartesian_data.append(
                [atom_types, atom_label, cartesian_positions])

        return cartesian_data

    def remove_duplicates_and_inverses(self, binding_sites):
        """
        Removes duplicate and inverse binding sites from a list of binding sites.

        Args:
            self (object): The object calling this method.
            binding_sites (list): A list of binding sites, where each binding site is represented
                as a list. Binding sites can have lengths of 1, 2, or 3.

        Returns:
            list: A list of binding sites with duplicates and inverses removed. The order of the
                original binding sites is preserved.
        """
        unique_sites = []
        for site in reversed(binding_sites):
            if len(site) == 1:
                unique_sites.append(site)
            elif len(site) == 2:
                if not any(np.array_equal(np.abs(site[1][1]), np.abs(unique_site[1][1])) for unique_site in unique_sites):
                    unique_sites.append(site)
            elif len(site) == 3:
                reversed_site = [(site[0]), (site[2]), (site[1])]
                if not any(np.array_equal(np.abs(reversed_site[1][1]), np.abs(unique_site[1][1])) and
                           np.array_equal(np.abs(reversed_site[2][1]), np.abs(unique_site[2][1])) for unique_site in unique_sites):
                    unique_sites.append(site)
        return list(reversed(unique_sites))
    

    def remove_duplicates(self, lst):
        """
        Remove duplicate sublists from a list based on the uniqueness of the third element 
        (numpy arrays) in each sublist.

        Parameters:
        - lst (list): A list of sublists, where each sublist contains two lists of strings 
                      and a list of numpy arrays as its third element.

        Returns:
        - list: A new list with duplicate sublists removed, determined by the uniqueness 
                of the numpy arrays in each sublist.
        """
        seen = set()
        return [
            seen.add(tuple(map(tuple, item[2]))) or item 
            for item in lst 
            if tuple(map(tuple, item[2])) not in seen
        ]

    def aligned_to(self, target, align=None, orient=None):
        """
        Aligns the guest positions to the given target position and optionally
        to align and orient positions.

        Args:
            target (tuple): A tuple containing (index, position) of the target position.
            align (tuple, optional): A tuple containing (index, position) to align with.
                                     Defaults to None.
            orient (tuple, optional): A tuple containing (index, position) to orient with.
                                      Defaults to None.

        Returns:
            list: A list of fractional coordinates representing the aligned and oriented guest positions.
        """

        get_struct = self.get_cube_structure
        target_idx, target = target[0], target[1]
        target_guest = self.cart_coords[target_idx]
        guest_position = [[x - y for x, y in zip(atom, target_guest)]
                          for atom in self.guest_molecule_data.cart_coords]

        if align is not None:
            align_idx, align = align

            align_guest = guest_position[align_idx]
            rotate = self.matrix_rotate(align_guest, align)
            guest_position = [np.dot(rotate, pos) for pos in guest_position]

            if orient is not None:
                orient_idx, orient = orient
                # guest position has changed
                align_guest = guest_position[align_idx]
                orient_guest = guest_position[orient_idx]
                # align normals
                normal_guest = np.cross(align_guest, orient_guest)
                normal_orient = np.cross(align, orient)
                # normals are [0.0, 0.0, 0.0] for parallel
                if normal_guest.any() and normal_orient.any():
                    rotate = self.matrix_rotate(normal_guest, normal_orient)
                    guest_position = [np.dot(rotate, pos)
                                      for pos in guest_position]

        # move to location after orientation as it is easier
        guest_position = [[x + y for x, y in zip(atom, target)]
                          for atom in guest_position]

        return [get_struct.lattice.get_fractional_coords(x) for x in guest_position]

    def matrix_rotate(self, source, target):
        """
        Create a rotation matrix that will rotate the source vector onto the target vector.

        Args:
            source (numpy.array): The source vector.
            target (numpy.array): The target vector.

        Returns:
            numpy.array: The 3x3 rotation matrix that aligns the source vector with the target vector.
        """
        source = np.asarray(source)/np.linalg.norm(source)
        target = np.asarray(target)/np.linalg.norm(target)
        v = np.cross(source, target)
        vlen = np.dot(v, v)
        if vlen == 0.0:
            return np.identity(3)
        c = np.dot(source, target)
        h = (1 - c)/vlen
        return np.array([[c + h*v[0]*v[0], h*v[0]*v[1] - v[2], h*v[0]*v[2] + v[1]],
                        [h*v[0]*v[1] + v[2], c + h*v[1]*v[1], h*v[1]*v[2] - v[0]],
                        [h*v[0]*v[2] - v[1], h*v[1]*v[2] + v[0], c + h*v[2]*v[2]]])
    
    def write_binding_sites_fractional(self, filename):
        cell_vectors = self.get_cube_structure.lattice.matrix

        with open(os.path.join(self.gala.output_directory, filename), 'w') as f:
            # Writing cell vectors to file
            f.write(f"{self.structure_name}\n")
            f.write('Cell_Lattice\n')

            for vector in cell_vectors:
                f.write("     %12.12f %12.12f %12.12f\n" % tuple(vector))

            # Writing binding sites to file
            for idx, bind in enumerate(self._binding_sites):
                this_point = ["BS: %i\n" % idx]
                for atom, label, coords in zip(bind[0], bind[1], bind[2]):
                    this_point.append("%-5s %-5s" % (atom, label))
                    this_point.append(
                        "%12.6f %12.6f %12.6f\n" % tuple(coords))
                f.writelines(this_point)


    def write_binding_sites(self, filename):
        """
        Writes the binding sites to a CIF file.

        Args:
            filename (str): The name of the output CIF file.

        Raises:
            Exception: If the binding sites are not calculated yet.
        """
        if self._binding_sites is None:
            self.calculate_binding_sites()
        structure_with_sites = self.get_cube_structure.copy()
        for binding_site in self._binding_sites:
            for atom, coord in zip(binding_site[0], binding_site[2]):
                if atom != str('D'):
                    structure_with_sites.append(species=atom,
                                                coords=coord)
                else:
                    pass
        self.structure_with_sites = structure_with_sites
        CifWriter(structure_with_sites).write_file(
            os.path.join(self.gala.output_directory, filename))

    def mk_dl_poly_control(self, cutoff, dummy=False):
        """CONTROL file for binding site energy calculation."""
        if dummy:
            stats = 1
            steps = 1
        else:
            stats = 1
            steps = 1
        control = [
            "# minimisation\n",
            "optim energy 1.0\n",  # gives approx 0.01 kcal
            "steps %i\n" % steps,
            "timestep 0.001 ps\n",
            "#ensemble nvt hoover 0.1\n",
            "cutoff %f angstrom\n" % cutoff,
            "delr 1.0 angstrom\n",
            "ewald precision 1d-6\n",
            "job time 199990 seconds\n",
            "close time 2000 seconds\n",
            "stats  %i\n" % stats,
            "#traj 1,100,2\n"
            "finish\n"]

        return control

    def try_symlink(self, src, dest):
        """Delete an existing dest file, symlink a new one if possible."""
        if os.path.lexists(dest):
            os.remove(dest)
        # Catch cases like Windows which can't symlink or unsupported SAMBA
        try:
            os.symlink(src, dest)
        except (AttributeError, OSError):
            shutil.copy(src, dest)

    def mk_dl_poly_config(self, formula, lattice_vectors, site_info, directory):
        """
        Create a DL_POLY CONFIG file based on the provided information.

        Args:
            formula (str): The chemical formula of the system.
            lattice_vectors (list): List of 3x3 arrays representing lattice vectors.
            site_info (list): List of site information, including atom types and coordinates.
            directory (str): The directory where the CONFIG file will be saved.

        Returns:
            None
        """
        output_file = os.path.join(directory, "CONFIG")
        header = formula
        atom_count = len(site_info)

        lattice_lines = [
            f"{lattice_vectors[i][0]:20.15f}{lattice_vectors[i][1]:20.15f}{lattice_vectors[i][2]:20.15f}" for i in range(3)]

        site_lines = []
        for atom in site_info:
            site_lines.append(
                f"{atom[1]:<6}{atom[0]:6d}\n{atom[2]:20.12f}{atom[3]:20.12f}{atom[4]:20.12f}")

        content = [
            header,
            f"         0         3{atom_count:10d}",
            *lattice_lines,
            *site_lines
        ]

        with open(output_file, 'w') as file:
            file.write('\n'.join(content))

    def split_and_recreate_supercell(self, supercell_data, grid_factors):
        """
        Split and recreate a supercell structure based on grid factors.

        Args:
            supercell_data: Data of the supercell structure.
            grid_factors (list): List of factors for splitting the supercell.

        Returns:
            supercell_structure: The recreated supercell structure.
        """
        total_cells = 1
        for factor in grid_factors:
            total_cells *= factor

        cells = [dict() for _ in range(total_cells)]

        for i, cell_data in enumerate(supercell_data):
            cell_num = i % total_cells
            cells[cell_num].setdefault('sites', []).append(cell_data)

        supercell_sites = []
        for cell_data in cells:
            cell_sites = cell_data.get('sites', [])
            supercell_sites.extend(cell_sites)

        supercell_structure = Structure.from_sites(supercell_sites)

        return supercell_structure

    def get_molecular_types_number(self, file_path):
        """
        Extract the number of molecular types from a file.

        Args:
            file_path (str): The path to the file to extract data from.

        Returns:
            int or None: The number of molecular types if found, or None if not found.
        """
        target_phrase = 'molecular types'
        with open(file_path, 'r') as file:
            for line in file:
                if target_phrase in line:
                    return int(line.split()[-1])
        return None

    def read_block(self, file):
        """Function to read blocks of data"""
        lines = []
        while True:
            line = file.readline()
            if not line or line.startswith("finish"):
                lines.append(line)
                break
            lines.append(line)
        return lines

    def extract_guest(self, file):
        """Function to extract guest data and its elements/sites"""
        guest_header = file.readline()
        guest_elements = []
        guest_data = [guest_header]
        block_data = self.read_block(file)
        for line in block_data:
            elements = line.split()
            if elements:
                guest_elements.append(elements[0])
        guest_data.extend(block_data)
        return guest_data, guest_elements

    def extract_framework(self, file):
        """Function to extract framework data and its elements/sites"""
        framework_header = file.readline()
        framework_elements = []
        framework_data = [framework_header]
        block_data = self.read_block(file)
        for line in block_data:
            elements = line.split()
            if elements:
                framework_elements.append(elements[0])
        framework_data.extend(block_data)
        return framework_data, framework_elements

    def extract_vdw_interactions(self, file, valid_elements):
        """Function to extract VDW interactions for only given valid elements"""
        vdw_header = file.readline().split()

        # Check if the header starts with VDW
        if not vdw_header or vdw_header[0] != "VDW":
            raise ValueError("Expecting a VDW header but not found")

        vdw_data = []
        vdw_count = 0
        while True:
            line = file.readline()
            if not line or line.startswith("finish"):
                break
            parts = line.split()
            if parts[0] in valid_elements and parts[1] in valid_elements:
                vdw_data.append(line)
                vdw_count += 1

        # Update the VDW header with the new count
        vdw_header = f"VDW {vdw_count}\n"

        # Return the header and the valid VDW interactions
        return [vdw_header] + vdw_data

    def create_field_files(self, desired_sites, directory):
        """
        Create FIELD files for DL_POLY based on desired binding sites.

        Args:
            desired_sites (list): List of desired binding sites.
            directory (str): The directory where the files will be created.

        Returns:
            None
        """
        with open(os.path.join(self.gala.directory, 'FIELD'), "r") as f:
            header = [f.readline(), f.readline()]
            molecular_types = int(f.readline().split()[2])
            guests = []

            # Extract guests' data
            for _ in range(molecular_types - 1):
                guest_data, guest_elements = self.extract_guest(f)
                # Filter based on desired sites
                if any(site in desired_sites for site in guest_elements):
                    guests.append((guest_data, guest_elements))
            framework_data, framework_elements = self.extract_framework(f)
            vdw_start_position = f.tell()

            for i, (guest_data, guest_elements) in enumerate(guests):
                valid_elements = set(guest_elements + framework_elements)
                f.seek(vdw_start_position)
                vdw_data = self.extract_vdw_interactions(f, valid_elements)
                with open(os.path.join(directory, 'FIELD'), "w") as out_file:
                    out_file.write("".join(header))
                    out_file.write("molecular types 2\n")
                    out_file.write("".join(guest_data))
                    out_file.write("".join(framework_data))
                    out_file.write("".join(vdw_data))
                    out_file.write("close\n")

    def modify_field_file(self, file_path, nummol_framework, is_empty):
        """
        Modify a DL_POLY FIELD file based on specified criteria.

        Args:
            file_path (str): The path to the FIELD file to modify.
            is_empty (bool): Whether to modify the FIELD file as empty or not.

        Returns:
            None
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        in_guest_section = False
        for index, line in enumerate(lines):
            if "&guest" in line:
                in_guest_section = True
                lines[index - 1] = "molecular types 2\n"
                lines[index + 1] = "NUMMOLS 1\n"
                continue

            if in_guest_section:
                if "ATOMS" in line:
                    # Skip the ATOMS line and modify the subsequent lines
                    atom_idx = index + 1
                    while not any(keyword in lines[atom_idx] for keyword in ["rigid", "Framework"]):
                        parts = lines[atom_idx].split()
                        if len(parts) > 3:
                            if is_empty == False:
                                parts = parts[:3] + ["1", "0"]
                            else:
                                parts = parts[:2] + ["0", "1", "0"]
                            lines[atom_idx] = "    ".join(parts) + "\n"
                        atom_idx += 1
                    in_guest_section = False
            
            if "Framework" in line:
                in_framework = True
                lines[index + 1] = f"NUMMOLS {nummol_framework}\n"
                continue

        with open(file_path, 'w') as f:
            f.writelines(lines)

    def _create_directory_structure(self, root_dir, sub_dir_name):
        """Create a directory structure and return the path."""
        dir_path = os.path.join(root_dir, sub_dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        return dir_path

    def _gather_site_info(self, binding_site):
        """Gather site information from a binding site."""
        site_info_list = []
        for idx, (atom, coord) in enumerate(zip(binding_site[1], binding_site[2])):
            site_info_list.append([idx+1, atom, coord[0], coord[1], coord[2]])
        return site_info_list, len(site_info_list) + 1

    def _handle_dl_poly_files(self, directory, binding_site_atoms, guest_idx, bs_idx, site_info_list, cell, number_of_cell_framework, standard):
        """
        Handle DL_POLY files for a specific directory and binding site.

        Args:
            directory (str): The directory to handle DL_POLY files in.
            binding_site_atoms (list): List of atoms in the binding site.
            guest_idx (int): Guest index.
            bs_idx (int): Binding site index.
            site_info_list (list): List of site information.
            cell (Structure): The supercell structure.
            cut (float): Cutoff for DL_POLY.

        Returns:
            None
        """
        cut = self.gala.md_cutoff
        field_path = os.path.join(self.gala.directory, 'FIELD')
        
        # Extract number of molecules
        num_value = 0
        with open(field_path, 'r') as file:
            found_framework = False
            for line in file:
                if found_framework and line.startswith('NUMMOLS'):
                    _, num_value = line.split()
                    num_value = int(num_value)
                    break
                elif line.strip() == 'Framework':
                    found_framework = True

        for idx, site in enumerate(cell.sites):
            element = site.species_string
            coords = site.coords
            site_info_list.append(
                [idx + guest_idx, element, coords[0], coords[1], coords[2]])

        self.mk_dl_poly_config(self.structure_name,
                               cell.lattice.matrix, site_info_list, directory)

        if bs_idx > 0:
            os.chdir(directory)
            # symlink on FIELD to save space
            zero_directory = "%s_bs_%04d" % (
                self.composition.reduced_formula, 0)
            self.try_symlink(os.path.join('..', zero_directory, 'FIELD'),
                             'FIELD')
            self.try_symlink(os.path.join('..', zero_directory, 'CONTROL'),
                             'CONTROL')
        elif os.path.isfile(field_path):
            if self.get_molecular_types_number(field_path) == 2:
                shutil.copy(field_path, os.path.join(directory, 'FIELD'))
                self.modify_field_file(os.path.join(
                    directory, 'FIELD'), number_of_cell_framework, is_empty=standard)
            else:
                self.create_field_files(binding_site_atoms, directory)
                self.modify_field_file(os.path.join(
                    directory, 'FIELD'), number_of_cell_framework, is_empty=standard)
            with open(os.path.join(directory, "CONTROL"), "w") as control:
                control.writelines(self.mk_dl_poly_control(cut))
        else:
            print('Error - FIELD file missing')

    def Make_Files(self):
        """
        Create DL_POLY input files for different binding sites and configurations.

        Returns:
            list: A list of individual directories created for DL_POLY simulations.
        """
        binding_sites = self._binding_sites_cart
        guest = self.composition.reduced_formula
        fold_factor = self.minimum_supercell(cell=self.get_cube_structure.lattice.matrix, cutoff=self.gala.md_cutoff)
        repeated_cell_number = np.prod(fold_factor)
        unitcell = self.get_cube_structure.copy()
        os.chdir(self.gala.directory)
        unitcell.make_supercell(scaling_matrix=fold_factor)
        supercell = unitcell

        supercell_dlp = self.split_and_recreate_supercell(
            supercell, fold_factor)
        root_directory = "DL_poly_BS"
        root_directory = self._create_directory_structure(
            self.gala.directory, root_directory)
        current_directory = os.getcwd()

        individual_directories = []

        # Handle No binding site files
        no_bs_directory_name = f'{self.structure_name}_{guest}'
        no_bs_directory = self._create_directory_structure(
            root_directory, no_bs_directory_name)
        individual_directories.append(no_bs_directory)
        binding_site = binding_sites[0]
        binding_site_ = binding_site.copy()
        arbitrary_location = [np.array([i, 0, 0]) for i in range(len(binding_site_[-1]))]
        binding_site_[-1] = arbitrary_location
        site_info_list = []
        site_info_list, guest_idx = self._gather_site_info(binding_site_)
        self._handle_dl_poly_files(
            no_bs_directory, binding_sites[0][1], guest_idx, 0, site_info_list, supercell_dlp, repeated_cell_number, standard=True)

        for bs_idx, binding_site in enumerate(binding_sites):
            bs_directory_name = "%s_bs_%04d" % (guest, bs_idx)
            bs_directory = self._create_directory_structure(
                root_directory, bs_directory_name)

            site_info_list, guest_idx = self._gather_site_info(binding_site)
            self._handle_dl_poly_files(
                bs_directory, binding_site[1], guest_idx, bs_idx, site_info_list, supercell_dlp, repeated_cell_number, standard=False)
            individual_directories.append(bs_directory)

            os.chdir(current_directory)

        return individual_directories

    def run_simulation(self, directory, EXE, starting_dir, gala_delete_files):
        if 'REVIVE' in gala_delete_files or '*_bs_*' in gala_delete_files:
            rm_line = 'rm REVIVE 2> /dev/null\n'
        else:
            rm_line = ''

        command = f"cd {directory} && {EXE} >> {os.path.join(starting_dir, 'DL_POLY.output')} && {rm_line}"
        process = subprocess.Popen(command, shell=True, executable='/bin/bash')
        process.wait()

        if process.returncode != 0:
            error_message = f"Error: {EXE} failed in {directory} directory with return code {process.returncode}.\n"
            print(error_message)

            with open(os.path.join(starting_dir, 'DL_POLY.output'), 'a') as f:
                f.write(error_message)

            return process.returncode

        if not os.path.exists(os.path.join(directory, 'STATIS')):
            error_message = f"Error: STATIS file not found in {directory} directory.\n"
            print(error_message)

            with open(os.path.join(starting_dir, 'DL_POLY.output'), 'a') as f:
                f.write(error_message)

            return 2

        return 0  # return code 0 means success


    def Submit_DL_Poly(self, no_submit, bs_directories, starting_dir, EXE, gala_delete_files):
        if no_submit:
            print('info: GALA input files generated; skipping job submission\n')
            return

        start_time = time.time()

        with multiprocessing.Pool(processes=self.gala.md_cpu) as pool:
            results = pool.starmap(self.run_simulation, [(dir, EXE, starting_dir, gala_delete_files) for dir in bs_directories])

        self.check_and_save(bs_directories, starting_dir)

        if all(res == 0 for res in results):
            print('DL Poly terminated normally')
        elif any(res == 1 for res in results):
            raise RuntimeError("Encountered an error: DL_POLY.X failed during execution.")
        elif any(res == 2 for res in results):
            raise RuntimeError("Encountered an error: DL_POLY.X completed but the STATIS file was not found.")

        error_file = os.path.join(starting_dir, 'DL_POLY.output')
        if os.path.getsize(error_file) == 0:
            os.remove(error_file)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"DL Poly Subprocesses Elapsed Time: {elapsed_time:.2f} seconds")


    def check_and_save(self, bs_directories, starting_dir):
        """
        For each directory in bs_directories, checks for the presence of file1.
        If file1 does not exist, checks for file2 and saves its last 5 lines to output_name.
        """
        for dir_name in bs_directories:
            statis_path = os.path.join(starting_dir, 'DL_poly_BS', dir_name, 'STATIS')
            output_path = os.path.join(starting_dir, 'DL_poly_BS', dir_name, 'OUTPUT')
            error_file_path = os.path.join(starting_dir, 'DL_POLY.output')

            if not os.path.exists(statis_path) and os.path.exists(output_path):
                with open(output_path, 'r') as output:
                    lines = output.readlines()[-4:]

                with open(error_file_path, 'a') as error_file:
                    error_file.write(os.path.basename(dir_name) + "\n")
                    error_file.write("\n".join([line.strip() for line in lines if line.strip()]) + "\n")
                
                return


    def minimum_image(self, atom1, atom2, box):
        """
        Return the minimum image coordinates of atom2 with respect to atom1.

        Args:
            atom1 (Structure): First atom.
            atom2 (Structure): Second atom.
            box (list): The box dimensions.

        Returns:
            list: Minimum image coordinates of atom2.
        """
        f_coa = atom1.site_properties['fractional'][:]
        f_cob = atom2.site_properties['fractional'][:]

        fdx = f_coa[0] - f_cob[0]
        if fdx < -0.5:
            f_cob[0] -= 1
        elif fdx > 0.5:
            f_cob[0] += 1
        fdy = f_coa[1] - f_cob[1]
        if fdy < -0.5:
            f_cob[1] -= 1
        elif fdy > 0.5:
            f_cob[1] += 1
        fdz = f_coa[2] - f_cob[2]
        if fdz < -0.5:
            f_cob[2] -= 1
        elif fdz > 0.5:
            f_cob[2] += 1
        new_b = [f_cob[0]*box[0][0] + f_cob[1]*box[1][0] + f_cob[2]*box[2][0],
                 f_cob[0]*box[0][1] + f_cob[1]*box[1][1] + f_cob[2]*box[2][1],
                 f_cob[0]*box[0][2] + f_cob[1]*box[1][2] + f_cob[2]*box[2][2]]
        return new_b

    def minimum_width(self, cell):
        """
        Calculate the shortest perpendicular distance within the cell.

        Args:
            cell (list): The cell matrix.

        Returns:
            float: The minimum width within the cell.
        """
        a_cross_b = np.cross(cell[0], cell[1])
        b_cross_c = np.cross(cell[1], cell[2])
        c_cross_a = np.cross(cell[2], cell[0])

        volume = np.dot(cell[0], b_cross_c)

        return volume / min(np.linalg.norm(b_cross_c), np.linalg.norm(c_cross_a), np.linalg.norm(a_cross_b))
    
    def minimum_supercell(self, cell, cutoff):
        """Calculate the smallest supercell with a half-cell width cutoff."""
        a_cross_b = np.cross(cell[0], cell[1])
        b_cross_c = np.cross(cell[1], cell[2])
        c_cross_a = np.cross(cell[2], cell[0])

        volume = np.dot(cell[0], b_cross_c)

        widths = [volume / np.linalg.norm(b_cross_c),
                  volume / np.linalg.norm(c_cross_a),
                  volume / np.linalg.norm(a_cross_b)]     

        return tuple(int(np.ceil(2*cutoff/x)) for x in widths)

    def update_gala(self, path):
        """
        Update structure properties from DL_POLY outputs.

        Args:
            cell (Structure): The cell structure.
            path (str): The path to the DL_POLY outputs.

        Returns:
            None
        """
        cell = self.get_cube_structure.lattice.matrix
        cell_data = [cell, self.minimum_width(cell), np.linalg.inv(cell.T)]
        self.gala_postproc(path, cell_data)

    def gala_postproc(self, filepath, cell_data):
        """
        Update structure properties from DL_POLY outputs.

        Args:
            filepath (str): The path to the file.
            test_folder (str): The test folder path.
            cell_data (list): List containing cell-related data.

        Returns:
            None
        """
        guest = self.composition.reduced_formula
        startdir = gala_input.directory
        os.chdir(filepath)
        
        if os.path.exists(os.path.join(filepath, f'{self.structure_name}_{guest}', 'STATIS')):
                statis = open(os.path.join(filepath, f'{self.structure_name}_{guest}', 'STATIS')).readlines()
                empty_esp = float(statis[3].split()[4])
        
        highest_occupency = self._binding_site_maxima[0][0][2]

        for directories in os.listdir():
            if directories.startswith(f'{guest}_bs'):
                binding_energies = []
                for bs_idx, binding_site in enumerate(self._binding_site_maxima):
                    bs_directory = "%s_bs_%04d" % (guest, bs_idx)
                    output = open(os.path.join(bs_directory,
                                               'OUTPUT')).readlines()

                    if 'error - quaternion integrator failed' in output[-1]:
                        # bad guest placement
                        e_vdw = float('nan')
                        e_esp = float('nan')
                        # put position of original as final position, nan anyway
                        try:
                            shutil.move(os.path.join(bs_directory, 'CONFIG'),
                                        os.path.join(bs_directory, 'REVCON'))
                        except IOError:
                            # CONFIG already moved, just doing update?
                            pass
                        # This is not a 'binding site'
                        magnitude = 0.0
                    else:
                        # energies
                        statis = open(os.path.join(bs_directory,
                                                   'STATIS')).readlines()
                        # Need to do a backwards search for the start of the last
                        # block since block size varies with number of atoms
                        # Timestep line will start with more spaces than data line
                        lidx = 0  # start block of final STATIS data
                        for ridx, line in enumerate(statis[::-1]):
                            if line.startswith('   ') and not 'NaN' in line:
                                lidx = ridx
                                break
                        else:
                            print('warning:No energies in STATIS\n')
                        e_vdw = float(statis[-lidx].split()[3])
                        e_esp = float(statis[-lidx].split()[4]) - empty_esp
                        # can just use the peak value
                        magnitude = binding_site[0][2]

                    # get position position
                    revcon = open(os.path.join(bs_directory,
                                               'REVCON')).readlines()
                    # If using MD, skip is 4 due to velocities and forces!
                    revcon = revcon[6::2]

                    # Fix molecule if it crosses boundaries
                    # have to make dummy objects with fractional attribute for
                    # the minimum_image function

                    # For now, put (atom, dummy) in here
                    positions = []

                    for atom, ratom in zip(self.guest_molecule_data.site_properties['elements'], revcon):
                        try:
                            dummy_pos = [float(x) for x in ratom.split()]
                        except ValueError:
                            # DL_POLY prints large numbers wrong, like
                            # 0.3970159038+105; these are too big anyway, so nan them
                            dummy_pos = [float('nan'), float(
                                'nan'), float('nan')]
                        # Sometimes atoms get flung out of the cell, but have
                        # high binding energy anyway, ignore them
                        if any((abs(x) > 2*cell_data[1]) for x in dummy_pos):
                            e_vdw = float('nan')
                            e_esp = float('nan')

                        dummy = self.guest_molecule_data.copy()

                        # create with the fractional, so position matches
                        dummy.add_site_property(
                            'fractional', [x % 1.0 for x in np.dot(cell_data[2], dummy_pos)])
                        self.guest_molecule_data.add_site_property('fractional', [x % 1.0 for x in np.dot(
                            cell_data[2], self.guest_molecule_data.cart_coords)])

                        # Put dummy nearest to first atom, if it exists
                        if positions:
                            dummy.add_site_property('pos', self.minimum_image(positions[0][1], dummy,
                                                    cell_data[0]))
                            positions.append((atom, dummy))
                        else:
                            # ensure anchor is within the unit cell
                            dummy.add_site_property('pos', np.dot(
                                dummy.site_properties['fractional'], cell_data[0]))
                            positions.append((atom, dummy))
                    # Extract the information we need from the tuples
                    positions = [(atom, dummy.site_properties['pos'])
                                 for atom, dummy in positions]
                    
                    percentage_occupancy = (magnitude / highest_occupency) * 100

                    # print("info:Binding site %i: %f kcal/mol, %.2f%% occupancy\n" %
                    #       (bs_idx, (e_vdw+e_esp), percentage_occupancy))
                    
                    binding_energies.append(
                        [percentage_occupancy, e_vdw, e_esp, positions])
                
                with open(os.path.join(self.gala.output_directory, '%s_gala_binding_sites.xyz' % guest), 'w') as gala_out:
                    frame_number = 0
                    for idx, bind in enumerate(binding_energies):
                        
                        energy = bind[1] + bind[2]
                        if energy > 0 or energy != energy:
                            continue

                        pc_elec = 100*bind[2]/energy
              
                        this_point = [
                            "%i\n" % len(bind[3]),  # number of atoms
                            # idx, energy, %esp, e_vdw, e_esp, magnitude
                            " BS: %i, Frame: %i, Ebind= %f, esp= %.2f%%, Evdw= %f, "
                            "Eesp= %.2f, occ= %.2f%%\n" %
                            (idx, frame_number, energy, pc_elec, bind[1], bind[2],
                             bind[0])]
                        for atom in bind[3]:
                            this_point.append("%-5s " % atom[0])
                            this_point.append(
                                "%12.6f %12.6f %12.6f\n" % tuple(atom[1]))
                        gala_out.writelines(this_point)
                        frame_number += 1
        
        os.chdir(startdir)


class GuestSites:

    all_structure = []

    def __init__(self, label, element, coords, charge, parent_molecule):
        """
        Initialize the Guest_Sites class with the specified label, element, coordinates, charge, and parent molecule.

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
        GuestSites.all_structure.append(self)
        self._structure = None
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
        if maxima_info is None:
            maxima_info = "We are unable to provide the maxima for this guest and its sites."
        return f"Site Label: {self.label}, Element: {self.element}, Coords: {self.coords}, Charge: {self.charge}\n{maxima_info}"

    @property
    def maxima(self):
        """
        Calculate and return the maxima information of the site.

        Returns:
            tuple: Tuple containing the fractional coordinates of the maxima and the maxima values.
        """
        if self.gala.method == 'FASTMC':

            if self.cube is None or self._datapoints is None:
                self.load_cube_data_fastmc()

            if self._maxima_coordinates is None or self._maxima_values is None:
                self.calculate_maxima()

            maxima_coords, maxima_values = self._maxima_coordinates, self._maxima_values
            
            if maxima_coords == ['N/A']:
                fractional_coords = ['N/A']
            elif maxima_coords is not None:
                fractional_coords = self.maxima_fractional_coordinates
            else:
                raise Exception(
                    'Maxima coordinates not calculated. Run calculate_maxima first.')

            maxima_info = "Maxima:\n"
            for i, (coords, value) in enumerate(zip(maxima_coords, maxima_values), start=1):
                if coords != 'N/A':
                    maxima_info += f"\tMaxima {i}: Value: {value}\n\t\t  Cartesian Coordinates: {list(coords)}\n\t\t  Fractional Coordinates: {list(fractional_coords[i-1])}\n"
                else: 
                    maxima_info += f"\tMaxima {i}: Value: {value}\n\t\t  Cartesian Coordinates: {coords}\n\t\t  Fractional Coordinates: {fractional_coords[i-1]}\n"
            
            return maxima_info

        else:
            pass

    @property
    def maxima_fractional_coordinates(self):
        """
        Calculate and return the fractional coordinates of the maxima.

        Returns:
            list: Fractional coordinates of the maxima.
        """
        if self._maxima_coordinates is None:
            self.calculate_maxima()

        if self.cube is None:
            if self._datapoints is None or self.cube is None:
                if self.gala.method == "FASTMC":
                    self.load_cube_data_fastmc()
                else:
                    raise ValueError(
                        "Invalid directory value: " + self.gala.method)

        lattice = self.cube.structure.lattice

        fractional_coordinates = [lattice.get_fractional_coords(
            cart_coords) for cart_coords in self._maxima_coordinates]

        return fractional_coordinates

    @property
    def maxima_cartesian_coordinates(self):
        """
        Calculate and return the fractional coordinates of the maxima.

        Returns:
            list: Fractional coordinates of the maxima.
        """
        if self._maxima_coordinates is None:
            self.calculate_maxima()

        if self.cube is None:
            self.load_cube_data_fastmc()

        return self._maxima_coordinates

    @property
    def structure(self):
        """
        Get the structure associated with the cube data.

        If the cube data is not loaded, it loads the data using FASTMC method

        Returns:
            pymatgen.Structure: The structure object associated with the cube data.
        """
        if self.cube is None:
            if self.gala.method == 'FASTMC':
                self.load_cube_data_fastmc()
        if self._maxima_coordinates == ['N/A']:
            if GuestSites.all_structure:
                self._structure = GuestSites.all_structure[0].cube.structure
        else:
            self._structure = self.cube.structure

        return self._structure

    def load_cube_data_fastmc(self):
        """
        Load the cube data for the site.
        """
        dir = self.gala.directory
        guests = self.parent_molecule.composition.reduced_formula
        sites = self.label
        sites_element = self.element
        fold = self.gala.gcmc_grid_factor

        probability_file = f'{dir}/Prob_Guest_{guests}_Site_{sites}_folded.cube'
        probability_file_unfolded = f'{dir}/Prob_Guest_{guests}_Site_{sites}.cube'

        if os.path.exists(probability_file):
            try:
                self.cube = VolumetricData.from_cube(probability_file)
                localdata = self.cube.data['total']
                localdata = localdata / np.sum(localdata)
                if self.gala.gcmc_write_folded:
                    print('Input Files Were Already Folded, Skipped')
                self._datapoints = localdata
            except Exception as e:
                raise Exception(f'Error loading cube file: {probability_file}')
            
        elif not os.path.exists(probability_file) and os.path.exists(probability_file_unfolded):
            try:
                self.cube = VolumetricData.from_cube(probability_file_unfolded)
                # TODO Write Check to see if division is ok (must dim must be divisible by fold)
                folded_dim = self.cube.dim // np.array(fold)
                unit_cell = self.get_unit_cell_structure(fold)
                localdata = self.folding_cube_file(fold, folded_dim)
                localdata = localdata / np.sum(localdata)
                self.cube = VolumetricData(unit_cell, {'total': localdata})
                if self.gala.gcmc_write_folded:
                    self.save_folded_cube(dir, guests, sites)
                self._datapoints = localdata
                
            except Exception as e:
                raise Exception(
                    f'Error loading cube file: {probability_file_unfolded}')
        
        elif not os.path.exists(probability_file) and sites_element == 'D':
            self._maxima_coordinates = ['N/A']; self._maxima_values = ['N/A']

        else:
            raise FileNotFoundError('''Error Code: UNAVAILABLE_CUBE_FILE
Error Description: Unavailable Probability File
Error Message: We apologize, but we couldn't locate any available unfolded or folded probability file required for this operation.\n''')

    def get_unit_cell_structure(self, fold):
        """
        Computes the unit cell structure by considering a subset of the structure 
        within the fractional coordinate bounds, determined by the given fold.
        
        Parameters:
        - fold (tuple): A tuple of integers representing the fold in each 
          x, y, z dimension respectively.

        Returns:
        - unit_cell (Structure): A Structure object representing the unit cell, 
          which contains a subset of sites from the original structure based on 
          the fold, and has a lattice rescaled accordingly.

        Example:
        - If the fold is (2,2,2), it means we are considering half of the structure 
          in each dimension to compute the unit cell structure.
        """
        bounds = (1 / fold[0], 1 / fold[1], 1 / fold[2])
        unit_cell_sites = [site for site in self.cube.structure if all(0 <= c < bounds[i] for i, c in enumerate(site.frac_coords))]
        unit_cell_sites_frac_coords = [site.frac_coords * fold for site in unit_cell_sites]

        constraint_lattice = self.cube.structure.lattice.matrix / np.array(fold)[:, None]
        unit_cell = Structure(constraint_lattice,
                            [site.species_string for site in unit_cell_sites],
                            unit_cell_sites_frac_coords,
                            coords_are_cartesian=False)
        return unit_cell


    def folding_cube_file(self, fold, folded_dim):
        """
        Folds the cube file data by averaging grid points over blocks defined 
        by the fold, and then normalizes the folded data.

        Parameters:
        - fold (tuple): A tuple of integers indicating how many times the 
          original data should be folded in each dimension (x, y, z).
        - folded_dim (tuple): A tuple indicating the dimensions of the folded 
          data cube.

        Returns:
        - normalized_data (ndarray): A numpy array containing the averaged and 
          normalized data after folding according to the specified fold.

        Example:
        - If the fold is (2,2,2), the original data will be divided into 8 blocks, 
          and the values within each block are averaged to obtain the folded data.
        """
        localdata = np.zeros((fold[0] * fold[1] * fold[2], folded_dim[0], folded_dim[1], folded_dim[2]))
        for xidx in range(fold[0]):
            for yidx in range(fold[1]):
                for zidx in range(fold[2]):
                    grid_idx = zidx + yidx * fold[2] + xidx * fold[2] * fold[1]
                    localdata[grid_idx] = self.cube.data['total'][
                                        (xidx * self.cube.dim[0]) // fold[0]:((xidx + 1) * self.cube.dim[0]) // fold[0],
                                        (yidx * self.cube.dim[1]) // fold[1]:((yidx + 1) * self.cube.dim[1]) // fold[1],
                                        (zidx * self.cube.dim[2]) // fold[2]:((zidx + 1) * self.cube.dim[2]) // fold[2]]

        avg_data = np.mean(localdata, axis=0)
        normalized_data = avg_data / np.sum(avg_data)
        return normalized_data


    def save_folded_cube(self, dir, guests, sites):
        """
        Saves the folded cube data to a file.

        Parameters:
        - dir (str): The directory where the folded cube file will be saved.
        - guests (str): A string representing the guests parameter, 
          to be included in the filename.
        - sites (str): A string representing the sites parameter, 
          to be included in the filename.

        Returns:
        - None

        Side Effect:
        - A new .cube file is created in the specified directory with the 
          folded data.

        Example filename:
        - 'Prob_Guest_<guests>_Site_<sites>_folded.cube', where <guests> and <sites> 
          will be replaced by the actual values passed to the function.
        """
        file_path = os.path.join(dir, f'Prob_Guest_{guests}_Site_{sites}_folded.cube')
        self.cube.to_cube(file_path)
    
    def calculate_maxima(self):
        """
        Calculate the maxima for the site.
        """
        if self._datapoints is None or self.cube is None:
            if self.gala.method == "FASTMC":
                self.load_cube_data_fastmc()
            else:
                raise ValueError(
                    "Invalid directory value: " + self.gala.method)

        original_data = self._datapoints
        temp_data = original_data
        normalising_sum = sum(temp_data)
        dimension = (np.array(self.cube.dim)).reshape(-1, 1)
        cell = self.cube.structure.lattice.matrix
        spacing = np.linalg.norm(cell[0][0] / dimension[0])
        cell_total = cell / dimension

        sigma = (self.gala.gcmc_sigma / spacing) ** 0.5
        temp_data = gaussian_filter(temp_data, sigma, mode="wrap")

        temp_data *= normalising_sum / sum(temp_data)

        if self.gala.gcmc_write_smoothed:
            guests = self.parent_molecule.composition.reduced_formula
            sites = self.label
            VolumetricData(self.cube.structure, {'total': temp_data}).to_cube(os.path.join(self.gala.directory, f'Prob_Guest_{guests}_Site_{sites}_smoothed.cube'))

        neighborhood = generate_binary_structure(np.ndim(temp_data), 2)

        footprint = int(round(self.gala.gcmc_radius / spacing, 0))
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
            if point[1] > self.gala.gcmc_cutoff * maximum_value:
                pruned_peaks.append(point)

        self._maxima_coordinates = [point[0] for point in pruned_peaks]
        self._maxima_values = [point[1] for point in pruned_peaks]  


def post_gala_cleaning(dir):
    if os.path.exists(os.path.join(dir, 'DL_poly_BS')):
        shutil.rmtree(os.path.join(dir, 'DL_poly_BS'))


if __name__ == "__main__":

    # --- Timing ---
    start_time = time.time()
    # --- --- ---

    GALA_MAIN = os.getcwd()
    gala_input = GalaInput(GALA_MAIN)

    # Following section was for testing. when running the code, it will remove the DL_poly_BS folder and recreate all the directories.
    if os.path.exists(os.path.join(gala_input.directory, 'DL_poly_BS')) and os.path.isdir(os.path.join(gala_input.directory, 'DL_poly_BS')):
        shutil.rmtree(os.path.join(gala_input.directory, 'DL_poly_BS'))
    else:
        pass
    
    

    structure = GuestStructure(gala_input)
    for i in range(len(structure.guest_molecules)):
        composition_formula = structure.guest_molecules[i].composition.reduced_formula
        filename_cif = f"{composition_formula}_binding_sites.cif"
        filename_xyz = f"{composition_formula}_binding_sites_fractional.xyz"
        structure.guest_molecules[i].write_binding_sites(filename=filename_cif)
        structure.guest_molecules[i].write_binding_sites_fractional(filename=filename_xyz)
        directories = structure.guest_molecules[i].Make_Files()

        structure.guest_molecules[i].Submit_DL_Poly(
            False, 
            directories, 
            gala_input.directory,
            gala_input.md_exe,
            ['REVIVE']
        )

        structure.guest_molecules[i].update_gala(
            os.path.join(gala_input.directory, 'DL_poly_BS'))

    if gala_input.cleaning:
        post_gala_cleaning(gala_input.directory)

    # --- --- ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time before completion: {elapsed_time:.2f} seconds")
    # --- --- ---
