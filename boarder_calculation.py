import numpy as np

def boarder_calculation_polar_coord_around_star(star_rg, r_array, theta_array, d_r, d_theta, rho_trshld):

    n_theta = int(2 * np.pi / d_theta)

    THETA = np.linspace(0, 2 * np.pi, n_theta)
    d_theta = THETA[1] - THETA[0]
    
    R = np.ones_like(THETA) * star_rg
    for i , min_angle in enumerate(np.linspace(0, 2 * np.pi, n_theta)[:-1]):
        max_angle = np.linspace(0, 2 * np.pi, n_theta)[i+1]
        f_r = r_array[(theta_array >= min_angle) & (theta_array <max_angle) ]
        print(min_angle, max_angle)
        print(f_r)
        f_f_r = f_r[f_r > star_rg]
        try:
            r_max = f_f_r.max()
        except:
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
        
    R[-1] = R[0]        
    return THETA + d_theta/2 , R