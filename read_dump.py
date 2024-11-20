import numpy as np

def load_dump(file_path, rescale = False):
    """
    Reads a LAMMPS dump file and organizes the data into a dictionary of timesteps, box bounds, and columns.
    
    Parameters:
    ----------
    file_path : str
        Path to the LAMMPS dump file.
    
    Returns:
    -------
    timesteps : dict
        Dictionary where keys are sequential timestep indices (0, 1, 2, ...) and values are numpy arrays
        containing the atomic data (id, type, x, y, z).
    box_bounds : numpy.ndarray
        Array containing the simulation box boundaries for x, y, and z dimensions.
    columns : list
        List of column names in the atom data (e.g., id, type, x, y, z).
    timestep_indices : list
        List of the actual timestep values as they appear in the dump file (e.g., [0, 500, 1000, ...]).
    
    Notes:
    -----
    - Atom positions are stored as 2D data (x, y), with z values set to 0.0 for compatibility in 2D systems.
    - The timestep keys in the `timesteps` dictionary are indexed sequentially starting from 0.
    - Use `timestep_indices` to map sequential indices back to the actual timestep values from the file.
    """
    timesteps = {}
    timestep_indices = []
    box_bounds = None
    columns = []
    
    with open(file_path, 'r') as file:
        timestep = None
        atom_data = []
        while True:
            line = file.readline()
            if not line:
                # End of file
                if timestep is not None:
                    timesteps[len(timestep_indices)] = np.array(atom_data)
                break
            if "ITEM: TIMESTEP" in line:
                # Store previous timestep data if it exists
                if timestep is not None:
                    timesteps[len(timestep_indices)] = np.array(atom_data)
                # Read new timestep and reset atom data
                timestep = int(file.readline().strip())
                timestep_indices.append(timestep)
                atom_data = []
            elif "ITEM: NUMBER OF ATOMS" in line:
                num_atoms = int(file.readline().strip())
            elif "ITEM: BOX BOUNDS" in line:
                # Read the next three lines as box bounds
                box_bounds = [list(map(float, file.readline().split())) for _ in range(3)]
                print(box_bounds)
            elif "ITEM: ATOMS" in line:
                # Column names are based on the line, expected to contain id, type, x, y, z
                columns = line.split()[2:]
            else:
                # Read atom data (id, type, x, y, z)
                data = line.split()
                if rescale:
                    atom_data.append(np.array([int(data[0]), int(data[1]), float(data[2]) * box_bounds[0][0] * 2, float(data[3])  * box_bounds[1][0] * 2, 0.0]))
                else:
                    atom_data.append(np.array([int(data[0]), int(data[1]), float(data[2]), float(data[3]), 0.0]))
        # Store last timestep if any
        if timestep is not None:
            timesteps[len(timestep_indices)] = np.array(atom_data)
    
    return timesteps, np.array(box_bounds), columns, timestep_indices


def get_star_info(timestep_trj, core_index, star_type, box_info):
    core = timestep_trj[timestep_trj[:,0] == core_index]
    core = np.column_stack((core[:, 2], core[:,3]))
    star_mono = timestep_trj[timestep_trj[:,1] == star_type]
    star_mono = np.column_stack((star_mono[:, 2], star_mono[:, 3]))

    star_mono -= core
    ## applying periodic boundary condition.
    
    # computing r:
    r = star_mono[:, 0] * star_mono[:, 0] + star_mono[:, 1] * star_mono[:, 1]
    r = np.sqrt(r)
    
    # computing theta
    theta = np.arccos(star_mono[:, 0]/ r)
    
    # computing rg
    star_mono -= np.mean(star_mono, axis = 0)

    rg = (np.sum(star_mono ** 2) / np.size(star_mono[:,0]))**0.5
    
    return rg, r, theta
    
    
    