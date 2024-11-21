import numpy as np

def boarder_calculation_polar_coord_around_star(star_rg, r_array, theta_array, d_r, d_theta, rho_trshld):
    """
    Computes the boundary of a star polymer in polar coordinates by analyzing 
    density distribution around the core. The boundary is determined where 
    density surpasses a given threshold.

    Parameters:
    ----------
    star_rg : float
        Radius of gyration of the star's monomers.
    r_array : np.ndarray
        Radial distances of monomers from the core.
    theta_array : np.ndarray
        Angular positions of monomers relative to the core.
    d_r : float
        Radial resolution for density evaluation.
    d_theta : float
        Angular resolution (radians) for density evaluation.
    rho_trshld : float
        Density threshold to determine the boundary.

    Returns:
    -------
    THETA : np.ndarray
        Array of angular positions (centered on each segment) in radians.
    R : np.ndarray
        Boundary radii corresponding to each angular segment.

    Steps:
    -----
    1. Divide 2π angular space into bins based on `d_theta`.
    2. For each angular segment:
       - Extract monomers within the angular bounds.
       - Identify the furthest monomer beyond the radius of gyration (`star_rg`).
       - Analyze the density of monomers in decreasing radial shells.
       - Mark the radial position where density first exceeds `rho_trshld`.
    3. Return angular positions and corresponding boundary radii.
    """
    n_theta = int(2 * np.pi / d_theta)

    THETA = np.linspace(0, 2 * np.pi, n_theta)
    d_theta = THETA[1] - THETA[0]
    
    R = np.ones_like(THETA) * star_rg
    for i , min_angle in enumerate(np.linspace(0, 2 * np.pi, n_theta)[:-1]):
        max_angle = np.linspace(0, 2 * np.pi, n_theta)[i+1]
        f_r = r_array[(theta_array >= min_angle) & (theta_array <max_angle) ]

        f_f_r = f_r[f_r > star_rg]
        try:
            r_max = f_f_r.max()
        except ValueError:
            r_max = star_rg
        for s in np.arange(r_max, star_rg, -d_r):
            d = np.size(f_f_r[(f_f_r < s + 0.5) & (f_f_r >= s)])
            if d>=1:
                area = d_theta * d_r * (2 * s - d_r)
                # print(d/area)
            if d/area> rho_trshld:
                # print(d/area)
                R[i] = s - d_r/2
                break
    # Ensure periodicity in the angular data
    R[-1] = R[0]        
    return THETA + d_theta/2 , R