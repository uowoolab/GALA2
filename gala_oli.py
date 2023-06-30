#!/usr/bin/env python3

import os
import periodictable


def GALA_input(dir):

    # Create an empty dictionary to store the variables
    GALA = {}

    with open(f'{dir}/GALA.inp', 'r') as f:
        l = f.readlines()
    f.close()

    # FASTMC or RESPA
    GALA["Method"] = l[27].upper().strip('\n')
    # Directory
    GALA["Directory"] = l[29].strip('\n')
    # GCMC Temperature
    GALA['Temperature'] = float(l[33].strip('\n'))
    # Cutoff (Angstrom)
    GALA['Cutoff_GCMC'] = float(l[35].strip('\n'))
    # Delr (Angstrom)
    GALA['Delr'] = float(l[37].strip('\n'))
    # Ewald Precision
    GALA['Ewald'] = l[39].strip('\n')
    # Grid Factors
    GALA['Grid_Factor'] = tuple(int(num)
                                for num in l[41].strip('[]\n').replace(' ', ''))
    # Grid Spacing
    GALA['Grid_Spacing'] = float(l[43].strip('\n'))
    # Sigma
    GALA['Sigma'] = float(l[47].strip('\n'))
    # Radius
    GALA['Radius'] = float(l[49].strip('\n'))
    # Cutoff
    GALA['Cutoff'] = float(l[51].strip('\n'))
    #  Write folded
    GALA['Write_Folded'] = l[53].strip('\n')
    #  Write Smoothed
    GALA['Write_Smoothed'] = l[55].strip('\n')
    #  Guest Molecule
    GALA['Selected_Guest'] = tuple(l[63].split())
    #  Sites
    GALA['Selected_site'] = tuple([group.split(',') if ',' in group else [
                                  group] for group in l[65].split()])

    # return the dict
    return GALA


def get_periodic_element(variable):
    element_symbols = [element.symbol for element in periodictable.elements]
    for symbol in element_symbols:
        if variable.startswith(symbol):
            return symbol
    return None


def get_molecule(elements_list):
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


def Parse_FASTMC_files(fastmc_dir):
    try:
        with open(os.path.join(fastmc_dir, "FIELD"), "r") as f:
            field_lines = f.readlines()
    except FileNotFoundError:
        print("Cannot find FIELD file...")

    struct_name = field_lines[0].strip("\n")

    fastmc_dict = {
        "name": struct_name,
        'guests_name': [],
        'guests': [],
        "guests_species": [],
        'atomic_weight': [],
        'coordinates': [],
        'charge': []
    }

    atom_count = 0
    read_atoms = False
    current_guest_atoms = []  # list to store current guest atoms
    current_guest_atom_type = []  # list to store current guest atoms type
    current_guest_atomic_weight = []  # list to store current guest atomic weights
    current_guest_coordinates = []  # list to store current guest coordinates
    current_guest_charge = []  # list to store current guest charge

    for line in field_lines:

        if "&guest" in line:
            if current_guest_atoms:  # if we have some atoms for previous guest
                # add the atoms to the guest list in dict
                fastmc_dict["guests_species"].append(current_guest_atoms)
                fastmc_dict['guests'].append(
                    ''.join(get_molecule(current_guest_atom_type)))
                fastmc_dict["atomic_weight"].append(
                    current_guest_atomic_weight)  # add atomic weights to dict
                fastmc_dict["coordinates"].append(
                    current_guest_coordinates)  # add coordinates to dict
                fastmc_dict["charge"].append(
                    current_guest_charge)  # add charges to dict
                # reset the lists for next guest
                current_guest_atoms = []
                current_guest_atom_type = []
                current_guest_atomic_weight = []
                current_guest_coordinates = []
                current_guest_charge = []
            guest_name = line.strip("&guest ").split(":")[0]
            fastmc_dict["guests_name"].append(guest_name)
            read_atoms = True

        elif line.startswith('ATOMS') and read_atoms == True:
            atom_count = int(line.split()[1])  # get the number of atoms

        elif atom_count > 0 and read_atoms == True:
            data = line.split()
            atom_label = data[0]
            atomic_weight = float(data[1])
            if atomic_weight != 0:
                element = get_periodic_element(atom_label)
                if element:
                    current_guest_atom_type.append(element)
                else:
                    pass

            x_coor = float(data[2])
            y_coor = float(data[3])
            z_coor = float(data[4])
            charge = float(data[5])
            # add the atom to the current guest atoms
            current_guest_atoms.append(atom_label)
            current_guest_atomic_weight.append(atomic_weight)
            # add the coordinates as a tuple
            current_guest_coordinates.append((x_coor, y_coor, z_coor))
            current_guest_charge.append(charge)
            atom_count -= 1

        elif line.startswith('Framework'):
            read_atoms = False  # stop reading atoms once we reach the Framework section
            if current_guest_atoms:  # if we have some atoms for current guest
                # add the atoms to the guest list in dict
                fastmc_dict["guests_species"].append(current_guest_atoms)
                fastmc_dict['guests'].append(
                    ''.join(get_molecule(current_guest_atom_type)))
                fastmc_dict["atomic_weight"].append(
                    current_guest_atomic_weight)  # add atomic weights to dict
                fastmc_dict["coordinates"].append(
                    current_guest_coordinates)  # add coordinates to dict
                fastmc_dict["charge"].append(
                    current_guest_charge)  # add charges to dict

    return fastmc_dict


def Parse_RASPA_files():
    # currently working on
    pass


def get_maxima(guest, site, coordinates):
    print(f'Guest: {guest}, Site: {site}, Coordinates: {coordinates}')


def check_guests_sites(GALA, fastmc_dict):
    # Initialize a boolean to keep track of the check status
    check_status = True

    # Check if all selected guests are in fastmc_dict['guests']
    missing_guests = [guest for guest in GALA['Selected_Guest']
                      if guest not in fastmc_dict['guests']]
    if missing_guests:
        print(
            f'Error: The following guests in the GALA dictionary are not found in the fastmc_dict: {missing_guests}')
        check_status = False

    # Flatten the list of sites in fastmc_dict['guests_species']
    fastmc_sites = [site for sublist in fastmc_dict['guests_species']
                    for site in sublist]
    # Check if all selected sites are in the flattened list of sites
    missing_sites = [site for site_group in GALA['Selected_site']
                     for site in site_group if site not in fastmc_sites]
    if missing_sites:
        print(
            f'Error: The following sites in the GALA dictionary are not found in the fastmc_dict: {missing_sites}')
        check_status = False

    return check_status


if __name__ == "__main__":

    GALA_MAIN = '/Users/oliviermarchand/Desktop/Python_Codes/Masters/GALA_v2/Jake/GALA-main'

    GALA = GALA_input(GALA_MAIN)

    print(GALA)
    # os.getcwd()

    if GALA['Method'] == 'FASTMC':

        fastmc_dict = Parse_FASTMC_files(GALA['Directory'])
        print(fastmc_dict)

    elif GALA['Method'] == 'RASPA':

        Parse_RASPA_files()

    else:
        print('Unsupported Method: Error, line 2 GALA.inp')

    # Use the check function
    if check_guests_sites(GALA, fastmc_dict):
        for guest in GALA['Selected_Guest']:
            guest_index = fastmc_dict['guests'].index(guest)
            for site in GALA['Selected_site']:
                for each_site in site:
                    indices = [i for i, x in enumerate(
                        fastmc_dict['guests_species'][guest_index]) if x == each_site]
                    for index in indices:
                        maxima = get_maxima(
                            guest, each_site, fastmc_dict['coordinates'][guest_index][index])
