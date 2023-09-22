#!/usr/bin/env python3

import os
import numpy as np
from pymatgen.core import Element, IMolecule, Molecule
from pymatgen.io.common import VolumetricData
from pymatgen.util.coord import all_distances
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion, iterate_structure
from pymatgen.symmetry.analyzer import PointGroupAnalyzer as PGA
from pymatgen.analysis.structure_matcher import StructureMatcher
from itertools import combinations
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.io.cif import CifWriter


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

        if len(self.gala.Selected_Guest) != len(self.gala.Selected_Site):
            raise ValueError(
                "Error Code: GUEST_SITE_MISMATCH\nError Description: The selected guest and site information is inconsistent.")

        if self.method == 'FASTMC':
            guests_chemical_structure, guests_atoms_coordinates, guest_sites_labels, site_data = self.parse_fastmc()
            self.guest_molecules = []

            for element, coordinate, molecule_site_data in zip(guests_chemical_structure, guests_atoms_coordinates, site_data):
                guest_molecule = GuestMolecule(
                    species=element, coords=coordinate, sites_data=molecule_site_data, gala_instance=self.gala)
                self.guest_molecules.append(guest_molecule)

        elif self.method == 'RASPA':
            guests_chemical_structure, guests_atoms_coordinates, guest_sites_labels, site_data = self.parse_raspa()
            self.guest_molecules = []

            for element, coordinate, molecule_site_data in zip(guests_chemical_structure, guests_atoms_coordinates, site_data):
                guest_molecule = GuestMolecule(
                    species=element, coords=coordinate, sites_data=molecule_site_data, gala_instance=self.gala)
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
        current_guest = reset_current_guest()  # initialize current_guest

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
        x_coor = float(data[3])
        y_coor = float(data[4])
        z_coor = float(data[5])
        charge = float(data[2])
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

    def parse_raspa(self):
        """
        Parse the RASPA simulation data, pseudo atoms data, and molecule data.

        Returns:
            tuple: Tuple containing parsed guest information (elements, coordinates, species, site_data).
        """
        # Use the appropriate path to your RASPA files
        simulation_data = GuestStructure.extract_simulation_data(
            os.path.join(self.gala.Directory, "simulation.input"))
        pseudo_atoms_data = GuestStructure.extract_pseudo_atoms_data(
            os.path.join(self.gala.Directory, "pseudo_atoms.def"))
        guest_dict = GuestStructure.extract_molecule_data(
            simulation_data, pseudo_atoms_data)

        guests_reduced, elements, dummy_atoms = self.post_process_guests(
            guest_dict)

        selected_guests = self.select_guests(
            guest_dict, guests_reduced, elements, dummy_atoms)

        return selected_guests

    def extract_simulation_data(filename):
        molecule_names = []
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('Component'):
                    molecule_name = line.split()[-1]
                    molecule_names.append([molecule_name])
        return molecule_names

    def extract_pseudo_atoms_data(filename):
        pseudo_atoms_data = {}
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith("#type"):
                    break
            for line in file:
                parts = line.split()
                if not parts:
                    continue
                atom_type = parts[0]
                chem = parts[3] if parts[3] != '-' else 'D'
                charge = float(parts[6])
                pseudo_atoms_data[atom_type] = {'chem': chem, 'charge': charge}
        return pseudo_atoms_data

    def extract_molecule_data(molecule_names, pseudo_atoms_data):
        result = []

        for molecule_name in molecule_names:
            molecule_data = {
                'valid_labels': [],
                'coordinates': [],
                'species': [],
                'site_data': []
            }

            molecule_file = f'/Users/oliviermarchand/Desktop/Python_Codes/Masters/GALA_v2/Jake/GALA-main/Oli_test/{molecule_name[0]}.def'
            atomic_data = []
            with open(molecule_file, 'r') as file:
                copy = False
                for line in file:
                    if line.strip() == '# atomic positions':
                        copy = True
                    elif line.strip() == '# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb':
                        copy = False
                    elif copy:
                        atomic_data.append(line.split())

            # Construct the required lists from the atomic data
            species = [data[1] for data in atomic_data]
            elements = [pseudo_atoms_data[specie]['chem']
                        for specie in species if pseudo_atoms_data[specie]['chem'] != 'D']
            coordinates = [(float(data[2]), float(data[3]), float(data[4]))
                           for data in atomic_data if pseudo_atoms_data[data[1]]['chem'] != 'D']
            charge = [pseudo_atoms_data[specie]['charge']
                      for specie in species]
            site_data = [[specie, pseudo_atoms_data[specie]['chem'], (float(data[2]), float(data[3]), float(
                data[4])), charge] for data, specie, charge in zip(atomic_data, species, charge)]

            # Append to molecule_data
            molecule_data['valid_labels'].extend(elements)
            molecule_data['coordinates'].extend(coordinates)
            molecule_data['species'].extend(species)
            molecule_data['site_data'].extend(site_data)

            # Append molecule_data to result
            result.append(molecule_data)

        return result


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
        self.pmg_molecule = IMolecule(species, coords)
        self.gala = gala_instance
        self.guest_sites = [Site(*site_data, parent_molecule=self)
                            for site_data in sites_data]
        self._binding_sites = None
        self._binding_site_maxima = None
        self.point_group = PGA(self.pmg_molecule).get_pointgroup()
        self.structure = self.get_structure()
        self.structure_with_sites = None

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

        if self._binding_sites == None:
            self.calculate_binding_sites()

        binding_site_coords = self._binding_sites
        return binding_site_coords

    def get_structure(self):

        struct_list = []

        for site in self.guest_sites:
            struct_list.append(site.structure)

        for pair in combinations(struct_list, 2):
            if not StructureMatcher().fit(pair[0], pair[1]):
                raise Exception("Structures in the cube files do not match!")
        
        return struct_list[0]

    def calculate_binding_sites(self):

        # TODO (Jake): reversible_guest calculation not robust -- should be based only on the given set of atoms
        # not for the entire molecule (I think?)

        # This function should set the self._binding_sites attribute
        
        #This works for iterating over atoms instead of sites
        guest_atom_distances = []

        # Consider only real atoms for determining site location
        # i.e., ignore dummy sites
        # First, need to determine which sites are closest to the COM. Place closest
        # site first.
        for idx, atom in enumerate(self.species):
            dist = np.linalg.norm(self.center_of_mass - self.cart_coords[idx])
            for site in self.guest_sites:
                if str(atom) == str(site.element):
                    guest_atom_distances.append((dist, idx, site.label))

        guest_atom_distances.sort()        

        # Now we are determining which sites to place at maxima in the distribution
        # Need the pairwise distances between sites
        if len(guest_atom_distances) == 1:
            # Only one site, use it
            distance_0_1 = None
            distance_0_2 = None
            distance_1_2 = None
            linear_guest = True
            reversible_guest = True
        
        elif len(guest_atom_distances) == 2:
            distance_0_1 = np.linalg.norm(self.cart_coords[guest_atom_distances[0][1]] - 
                                          self.cart_coords[guest_atom_distances[1][1]])
            distance_0_2 = None
            distance_1_2 = None
            linear_guest = True
            reversible_guest = True if len(PGA(self.pmg_molecule).get_equivalent_atoms()["eq_sets"][1]) == 1 else False
        # If there are more than two atoms, take the closest three
        else:
            distance_0_1 = np.linalg.norm(self.cart_coords[guest_atom_distances[0][1]] - 
                                          self.cart_coords[guest_atom_distances[1][1]])
            distance_0_2 = np.linalg.norm(self.cart_coords[guest_atom_distances[0][1]] -
                                          self.cart_coords[guest_atom_distances[2][1]])
            distance_1_2 = np.linalg.norm(self.cart_coords[guest_atom_distances[1][1]] -
                                          self.cart_coords[guest_atom_distances[2][1]])
            linear_guest = True if str(self.point_group) == ("D*h" or "C*v") else False
            # If the non-central atom has another equivalent atom
            reversible_guest = True if len(PGA(self.pmg_molecule).get_equivalent_atoms()["eq_sets"][1]) == 2 else False

        overlap_tol = 0.3

        binding_sites = []

        # Atom closest to the COM (e.g., Cx or Nx)
        origin_key = guest_atom_distances[0][2]

        # Keep list of sets of atoms that are already included
        # so we don't use them twice
        reversible_sets = []

        #for origin_atom in sorted(guest_locations[origin_key],
        #                          key=operator.itemgetter(1), reverse=True)
        
        # Working with a dict of {label: site}
        sites = {x.label: x for x in self.guest_sites}
        for central_max, central_max_value in zip(sites[origin_key].maxima_fractional_coordinates,
                                                         sites[origin_key]._maxima_values):
            # Single site guest, just place it at the maximum
            if distance_0_1 is None:
                binding_sites.append([(guest_atom_distances[0][1], central_max, central_max_value)])
                continue

            # If we have more than one atom to align
            align_key = guest_atom_distances[1][2]
            # Why are we choosing 999.9 as the comparison point lol?
            align_closest = (999.9, None)
            align_found = False
            for align_atom_max, align_atom_max_value in zip(sites[align_key].maxima_fractional_coordinates,
                                                            sites[align_key]._maxima_values):
                if np.all(align_atom_max == central_max):
                    continue
                # Distance vector and distance between central site and alignment site
                vector_0_1 = pbc_shortest_vectors(self.structure.lattice,
                                                  central_max,
                                                  align_atom_max)[0][0]
                separation_0_1 = np.linalg.norm(vector_0_1)
                # How much overlap is there between the sites according to plot vs expected
                align_overlap = abs(separation_0_1 - distance_0_1)
                if align_overlap < align_closest[0]:
                    align_closest = (align_overlap, align_atom_max)
                if align_overlap < overlap_tol:
                    align_found = True

                    # If we only have a two atom guest
                    if distance_0_2 is None:
                        if reversible_guest:
                            # Check if this set of maxima was already found
                            this_set = sorted([central_max.tolist(), align_atom_max.tolist()])
                            if this_set in reversible_sets:
                                continue
                            else:
                                # First tuple is the central atom idx, central_max, val
                                # Second tuple is the other atom idx, distance vector from
                                # the first site
                                binding_sites.append([
                                    (guest_atom_distances[0][1], central_max, central_max_value),
                                    (guest_atom_distances[1][1], vector_0_1)
                                ])
                                reversible_sets.append(this_set)
                                continue
                        # If not reversible, then this orientation is necessarily unique -- keep it
                        else:
                            binding_sites.append([
                                (guest_atom_distances[0][1], central_max, central_max_value),
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
                    for orient_atom_max, orient_atom_max_value in zip(sites[orient_key].maxima_fractional_coordinates,
                                                                      sites[orient_key]._maxima_values):
                        vector_0_2 = pbc_shortest_vectors(self.structure.lattice,
                                                          central_max,
                                                          orient_atom_max)[0][0]
                        separation_0_2 = np.linalg.norm(vector_0_2)
                        vector_1_2 = pbc_shortest_vectors(self.structure.lattice,
                                                          align_atom_max,
                                                          orient_atom_max)[0][0]
                        separation_1_2 = np.linalg.norm(vector_1_2)

                        overlap_0_2 = abs(separation_0_2 - distance_0_2)
                        overlap_1_2 = abs(separation_1_2 - distance_1_2)

                        # This is the new closest orientation
                        if overlap_0_2 + 0.5*overlap_1_2 < orient_closest[0]:
                            orient_closest = (overlap_0_2 + 0.5*overlap_1_2, orient_atom_max)

                        # If we find two aligning sites within tolerance (multiple of 2 since we are fitting two sites)
                        if overlap_0_2 < overlap_tol and overlap_1_2 < 2*overlap_tol:
                            found_site = True
                            if reversible_guest:
                                this_set = sorted([central_max.tolist(),
                                                   align_atom_max.tolist(),
                                                   orient_atom_max.tolist()])
                                if this_set in reversible_sets:
                                    continue
                                elif linear_guest:
                                    # Just add the two sites (no need for angular alignment)
                                    binding_sites.append([
                                        (guest_atom_distances[0][1],
                                         central_max,
                                         central_max_value),
                                        (guest_atom_distances[1][1], vector_0_1)])
                                    reversible_sets.append(this_set)
                                # If not linear, need to add all three site locations
                                else:
                                    binding_sites.append([
                                        (guest_atom_distances[0][1],
                                         central_max,
                                         central_max_value),
                                        (guest_atom_distances[1][1], vector_0_1),
                                        # NOTE: Also changed indexing here... (Jake)
                                        (guest_atom_distances[2][1], vector_0_2)])
                            elif linear_guest:
                                # Just add the two sites
                                binding_sites.append([
                                    (guest_atom_distances[0][1],
                                        central_max,
                                        central_max_value),
                                    (guest_atom_distances[1][1], vector_0_1)])
                            else:
                                # Add all three sites
                                binding_sites.append([
                                    (guest_atom_distances[0][1],
                                     central_max,
                                     central_max_value),
                                    (guest_atom_distances[1][1], vector_0_1),
                                    # NOTE (Jake): Also changed indexing here
                                    (guest_atom_distances[2][1], vector_0_2)])

                    # If still no site, do the best we can
                    if not found_site:
                        if linear_guest:
                            # Don't care about reversiblity since no third site found
                            # but can still make the guest with just two sites
                            binding_sites.append([(guest_atom_distances[0][1],
                                                   central_max,
                                                   central_max_value),
                                                  (guest_atom_distances[1][1],
                                                   vector_0_1)])
            else:
                if distance_0_2 is None and align_closest[0] > distance_0_1:
                    # Very isolated atom, not within 2 tol dists of any others
                    # treat as isolated point atom and still make a guest
                    binding_sites.append([(guest_atom_distances[0][1],
                                           central_max,
                                           central_max_value)])

        # Need a more robust implmentation of this...
        # (e.g., aligned_to function in old Guest.py class)

        clean_binding_sites = []
        binding_site_maxima = []

        for binding_site in binding_sites:
            for n, atom in enumerate(binding_site):
                if n == 0:
                    central_element = str(self.species[atom[0]])
                    max_value = atom[2]
                    central_coords = atom[1]
                    if len(guest_atom_distances) == 1:
                        clean_binding_sites.append([central_element, central_coords])
                        binding_site_maxima.append(max_value)
                if n == 1:
                    dist_vector = atom[1]
                    align_coords = self.structure.lattice.get_fractional_coords(self.structure.lattice.get_cartesian_coords(central_coords) + dist_vector)
                    align_element = str(self.species[atom[0]])
                    if len(guest_atom_distances) == 2:
                        clean_binding_sites.append([[central_element, align_element], [central_coords, align_coords]])
                    if len(guest_atom_distances) == 3 and linear_guest and reversible_guest:
                        orient_coords = self.structure.lattice.get_fractional_coords(self.structure.lattice.get_cartesian_coords(central_coords) - dist_vector)
                        orient_element = align_element
                        clean_binding_sites.append([[central_element, align_element, orient_element], [central_coords, align_coords, orient_coords]])
                if n == 2:
                    dist_vector = atom[1]
                    orient_coords = self.structure.lattice.get_fractional_coords(self.structure.lattice.get_cartesian_coords(central_coords) + dist_vector)
                    orient_element = str(self.species[atom[0]])
                    clean_binding_sites.append([[central_element, align_element, orient_element], [central_coords, align_coords, orient_coords]])

        self._binding_sites = clean_binding_sites
        self._binding_site_maxima = binding_site_maxima

    def write_binding_sites(self, filename):
        if self._binding_sites is None:
            self.calculate_binding_sites()
        structure_with_sites = self.structure.copy()
        for binding_site in self._binding_sites:
            for atom, coord in zip(binding_site[0], binding_site[1]):
                structure_with_sites.append(species=atom,
                                            coords=coord)
        self.structure_with_sites = structure_with_sites
        CifWriter(structure_with_sites).write_file(filename)

class Site:
    def __init__(self, label, element, coords, charge, parent_molecule):
        """
        Initialize the Site class with the specified label, element, coordinates, charge, and parent molecule.

        Args:
            label (str): Label of the site.
            element (str): Chemical element of the site.
            coords (tuple): Cartesian coordinates of the site.
            charge (float): Charge of the site.
            parent_molecule (GuestMolecule): Parent molecule of the site.
        """
        self.label = label
        self.element = element
        self.coords = coords
        self.charge = charge
        self.parent_molecule = parent_molecule
        self.gala = parent_molecule.gala
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

        fractional_coords = self.maxima_fractional_coordinates

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
            self.calculate_maxima()

        if self.cube is None:
            self.load_cube_data()

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
                                    
    @property
    def structure(self):
        if self.cube is None:
            self.load_cube_data()
        self._structure = self.cube.structure
        return self._structure

    def calculate_maxima(self):
        """
        Calculate the maxima for the site.
        """
        if self._datapoints is None or self.cube is None:
            self.load_cube_data()

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
        for point in sorted(cartesian_peaks, key=lambda k: -k[1], reverse=False):
            if point[1] > self.gala.Cutoff * maximum_value:
                pruned_peaks.append(point)

        self._maxima_coordinates = [point[0] for point in pruned_peaks]
        self._maxima_values = [point[1] for point in pruned_peaks]



if __name__ == "__main__":
    GALA_MAIN = os.getcwd()
    gala_input = GalaInput(GALA_MAIN)

    # Print all output for specifed Guests and Sites
    structure = GuestStructure(gala_input)
    #print(structure)

    # # Prints only maxima for guest N2 of Nx site
    structure.guest_molecules[0].write_binding_sites(filename="CO2_binding_sites.cif")
    structure.guest_molecules[1].write_binding_sites(filename="N2_binding_sites.cif")

    # # Print IMolecule center of mass of guest 0 (CO2)
    # print(structure.guest_molecules[0].center_of_mass)

    # # Print IMolecule centered molecule of guest 0 (N2)
    # print(structure.guest_molecules[1].get_centered_molecule)

    # # Print Specific Site Data, for example guest 0, site 1 (CO2, Cx)
    # print(structure.guest_molecules[0].guest_sites[0])

    # # Print Specific Site Data, for example guest 0, site 1 (N2, COM (D))
    # print(structure.guest_molecules[1].guest_sites[1])
