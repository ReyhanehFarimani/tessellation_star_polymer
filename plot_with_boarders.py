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
    
    # Define a fixed palette with at least 32 colors
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939", "#8ca252",
        "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39", "#e7ba52", "#e7cb94", "#843c39", "#ad494a",
        "#d6616b", "#e7969c", "#7b4173", "#a55194", "#ce6dbd", "#de9ed6", "#3182bd", "#6baed6"
    ]
    
    
    plt.rc("text", usetex=True)
    plt.rc(
    "text.latex",
    preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}",
    )
    plt.rc("font", family="serif", size=12)
    plt.rc("savefig", dpi=200)
    plt.rc("legend", loc="best", fontsize="large", fancybox=False, framealpha=0.0)
    plt.rc("lines", linewidth=1.0, markersize=4, markeredgewidth=0.5)
    
    # Adjust figure size dynamically based on the box dimensions
    box_x, box_y = box_info
    aspect_ratio = box_y / box_x
    plt.figure(figsize=(8, 8 * aspect_ratio))
    color = 0
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
            X -= np.round((X ) / box_x) * box_x
            Y -= np.round((Y ) / box_y) * box_y

                    
        X[-1] = X[0]
        Y[-1] = Y[0]
        # Smooth boundary using Cubic Spline
        t = np.linspace(0, 1, len(X))  # Parameter for spline interpolation
        spline_x = CubicSpline(t, X, bc_type='periodic')
        spline_y = CubicSpline(t, Y, bc_type='periodic')

        t_smooth = np.linspace(0, 1, 10000)  # Increased resolution for smooth boundary
        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)
        if periodic:
            x_smooth -= np.round((x_smooth) / box_x) * box_x
            y_smooth -= np.round((y_smooth) / box_y) * box_y
        # Plot star monomers with transparency to reduce visual clutter
        plt.scatter(ts[ts[:, 1] == starType][:, 2], ts[ts[:, 1] == starType][:, 3], alpha=0.15, label=f"Star {coreID}", c = color_palette[color])

        #removing lines breaked due to boundaries:
        right_x_smooth = x_smooth[x_smooth< -box_x/2] + box_x
        right_y_smooth = y_smooth[x_smooth< -box_x/2]
        
        left_x_smooth = x_smooth[x_smooth> box_x/2] - box_x
        left_y_smooth = y_smooth[x_smooth> box_x/2]
        
        up_x_smooth = x_smooth[y_smooth< -box_y/2] 
        up_y_smooth = y_smooth[y_smooth< -box_y/2] + box_y
        
        down_x_smooth = x_smooth[y_smooth> box_y/2] 
        down_y_smooth = y_smooth[y_smooth> box_y/2] - box_y
     
        try :
            plt.plot(right_x_smooth, right_y_smooth, linewidth=2.5, color = color_palette[color])
            plt.plot(left_x_smooth, left_y_smooth, linewidth=2.5, color = color_palette[color])
            
        except:
            pass
        try:            
            plt.plot(down_x_smooth, down_y_smooth, linewidth=2.5, color = color_palette[color])
            plt.plot(up_x_smooth, up_y_smooth, linewidth=2.5, color = color_palette[color])

        except:
            pass
        # Plot smooth boundary
        plt.plot(x_smooth, y_smooth, label=f"Boundary {starType - 1}", linewidth=2.0, color = color_palette[color])
        color +=1
    # Configure plot appearance
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Star Polymer Boundaries")
    # plt.legend(loc="best", fontsize=8)
    plt.axis([-box_x/2, box_x/2, -box_y/2, box_y/2])  # Ensure plot covers the entire simulation box
    # plt.grid(True, linestyle="--", alpha=0.7)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
