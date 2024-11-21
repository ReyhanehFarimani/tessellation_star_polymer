import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from read_dump import get_star_info
from boarder_calculation import boarder_calculation_polar_coord_around_star

def plotter(ts, list_of_star_core_ID, list_of_star_type, box_info, dr, d_theta, rho_trshld, periodic=True):
    """
    Plots the boundary of stars in a periodic system, ensuring smooth, closed boundaries.

    Parameters:
    ----------
    ts : np.ndarray
        Timestep data from the simulation (atomic positions and types).
    list_of_star_core_ID : list
        List of IDs for the star cores.
    list_of_star_type : list
        List of star types corresponding to the cores.
    box_info : tuple
        Size of the periodic simulation box (box_x, box_y).
    dr : float
        Radial resolution for boundary calculation.
    d_theta : float
        Angular resolution for boundary calculation.
    rho_trshld : float
        Density threshold for determining the boundary.
    periodic : bool, optional
        Whether to apply periodic boundary conditions. Default is True.
    """
    # Adjust figure size dynamically based on the box dimensions
    box_x, box_y = box_info
    aspect_ratio = box_y / box_x
    plt.figure(figsize=(8, 8 * aspect_ratio))
    
    for coreID, starType in zip(list_of_star_core_ID, list_of_star_type):
        
        # Extract core coordinates
        core_x = ts[ts[:, 0] == coreID][:, 2]
        core_y = ts[ts[:, 0] == coreID][:, 3]
        
        # Get star info and boundary
        rg, r, theta = get_star_info(ts, coreID, starType, box_info)
        THETA, R = boarder_calculation_polar_coord_around_star(rg, r, theta, dr, d_theta, rho_trshld)
        
        # Convert polar to Cartesian
        X = R * np.cos(THETA)
        Y = R * np.sin(THETA)
        
        # Add core position
        X += core_x
        Y += core_y
        
        if periodic:
            # Apply periodic boundaries to the coordinates
            X -= np.round((X + box_x / 2) / box_x) * box_x
            Y -= np.round((Y + box_y / 2) / box_y) * box_y
        
        # Smooth boundary using Cubic Spline
        t = np.linspace(0, 1, len(X))  # Parameter for spline interpolation
        spline_x = CubicSpline(t, X, bc_type='periodic')
        spline_y = CubicSpline(t, Y, bc_type='periodic')

        t_smooth = np.linspace(0, 1, 10000)  # Increased resolution for smooth boundary
        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)

        # Plot star monomers with transparency to reduce visual clutter
        plt.scatter(ts[ts[:, 1] == starType][:, 2], ts[ts[:, 1] == starType][:, 3], alpha=0.3, label=f"Star {coreID}")
        
        #removing lines breaked due to boundaries:
        r_smooth = np.zeros_like(x_smooth)
        r_smooth[:-1] = (x_smooth[1:] - x_smooth[:-1])**2 + (y_smooth[1:] - y_smooth[:-1])**2
        r_smooth[-1] = (x_smooth[-1] - x_smooth[0])**2 + (y_smooth[-1] - y_smooth[0])**2
        x_smooth = x_smooth[r_smooth<100]
        y_smooth = y_smooth[r_smooth<100]
        # Plot smooth boundary
        plt.plot(x_smooth, y_smooth, label=f"Boundary {starType - 1}", linewidth=1.2)

    # Configure plot appearance
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Star Polymer Boundaries")
    plt.legend(loc="best", fontsize=8)
    # plt.axis([-box_x, box_x, -box_y, box_y])  # Ensure plot covers the entire simulation box
    # plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
