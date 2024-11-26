from shapely.geometry import Polygon
import numpy as np
from scipy.interpolate import CubicSpline
from boarder_calculation import boarder_calculation_polar_coord_around_star
from read_dump import get_star_info

def unfold_polygon(polygon, box_x, box_y):
    """
    Generates periodic images of a polygon in a periodic box.

    Parameters:
    ----------
    polygon : shapely.geometry.Polygon
        The original polygon to be unfolded.
    box_x : float
        Width of the simulation box.
    box_y : float
        Height of the simulation box.

    Returns:
    -------
    polygons : list of shapely.geometry.Polygon
        List of periodic images of the polygon.
    """
    translations = [
        (0, 0),  # Original position
        (-box_x, 0), (box_x, 0),  # Left and right images
        (0, -box_y), (0, box_y),  # Top and bottom images
        (-box_x, -box_y), (-box_x, box_y),  # Corners
        (box_x, -box_y), (box_x, box_y)
    ]
    return [Polygon(np.array(polygon.exterior.coords) + np.array([tx, ty])) for tx, ty in translations]


def compute_strong_invasion(ts, list_of_star_core_ID, list_of_star_type, box_info, dr, d_theta, rho_trshld):
    """
    Computes the area of invasion (overlap) between the boundaries of star polymers,
    accounting for periodic boundary conditions.

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

    Returns:
    -------
    invasion_areas : list of tuples
        Each tuple contains (coreID_1, coreID_2, overlap_area) for two overlapping stars.
    """
    box_x, box_y = box_info

    x_star = []
    y_star = []
    polygons = []  # Store polygons for each star boundary

    # Compute smoothed boundaries for all stars
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

        # Close the boundary
        X[-1] = X[0]
        Y[-1] = Y[0]

        # Smooth boundary using Cubic Spline
        t = np.linspace(0, 1, len(X))
        spline_x = CubicSpline(t, X, bc_type='periodic')
        spline_y = CubicSpline(t, Y, bc_type='periodic')

        t_smooth = np.linspace(0, 1, 10000)
        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)

        x_star.append(x_smooth)
        y_star.append(y_smooth)

        # Create a polygon for the star boundary
        boundary_polygon = Polygon(np.column_stack((x_smooth, y_smooth)))
        polygons.append(boundary_polygon)

    # Compute the invasion areas (overlap between polygons with PBC)
    invasion_areas = []

    for i in range(len(polygons)):

        for j in range(i + 1, len(polygons)):
            # Generate unfolded polygons for both stars
            unfolded_i = unfold_polygon(polygons[i], box_x, box_y)
            unfolded_j = unfold_polygon(polygons[j], box_x, box_y)

            # Compute the maximum overlap area across all periodic images
            max_overlap_area = 0
            for poly_i in unfolded_i:
                for poly_j in unfolded_j:
                    overlap = poly_i.intersection(poly_j)
                    if not overlap.is_empty:
                        max_overlap_area = max(max_overlap_area, overlap.area)

            if max_overlap_area > 0:
                invasion_areas.append(max_overlap_area)

    return np.array(invasion_areas)

def compute_highest_strong_invasion(Rg, density):
    epsilon = 0.000001
    d = np.sqrt(density)
    alpha = np.arccos(d/2/Rg)
    S = 4 * Rg * Rg * (4 * alpha - np.sin(2 * alpha))
    if S == 0:
        return epsilon
    return S


def compute_surface_and_area(ts, list_of_star_core_ID, list_of_star_type, box_info, dr, d_theta, rho_trshld):
    """
    Computes the surface (perimeter) and area of star boundaries.

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

    Returns:
    -------
    total_perimeter : np.ndarray
        Perimeters of each star's boundary.
    total_area : np.ndarray
        Areas of each star's boundary.
    """
    
    
    
    box_x, box_y = box_info

    x_star = []
    y_star = []
    polygons = []  # Store polygons for each star boundary

    # Compute smoothed boundaries for all stars
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

        # Close the boundary
        X[-1] = X[0]
        Y[-1] = Y[0]

        # Smooth boundary using Cubic Spline
        t = np.linspace(0, 1, len(X))
        spline_x = CubicSpline(t, X, bc_type='periodic')
        spline_y = CubicSpline(t, Y, bc_type='periodic')

        t_smooth = np.linspace(0, 1, 10000)
        x_smooth = spline_x(t_smooth)
        y_smooth = spline_y(t_smooth)

        x_star.append(x_smooth)
        y_star.append(y_smooth)

        # Create a polygon for the star boundary
        boundary_polygon = Polygon(np.column_stack((x_smooth, y_smooth)))
        polygons.append(boundary_polygon)

    # Compute perimeters and areas
    total_perimeter = []
    total_area = []
    for polygon in polygons:
        total_perimeter.append(polygon.length)  # Compute perimeter
        total_area.append(polygon.area)  # Compute area

    return np.array(total_perimeter), np.array(total_area)
